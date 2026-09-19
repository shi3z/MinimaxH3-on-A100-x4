// FA3_SM80_H3 : FlashAttention on A100 (sm_80), hard-wired to MiniMax-H3 geometry.
// B=1, H=56, D=128, N variable, BF16, non-causal, no mask.
// v0: single-warp-group flash. cp.async K/V staging, mma.sync.m16n8k16 QK & PV,
//     fp32 accumulators, online softmax, register-resident Q.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

#define D        128
#define BM       64
#define BN       64
#define NWARPS   4
#define NTHREADS (NWARPS*32)
#ifndef STAGES
#define STAGES   2
#endif

__device__ __forceinline__ unsigned smem_u32(const void* p){
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// ldmatrix x4 (non-transposed): loads a 16x16 bf16 tile starting at (base) with
// element row-stride `stride`, into the 4 mma A-operand registers.
__device__ __forceinline__ void ldm_x4(uint32_t r[4], const bf16* base, int stride){
    int lane = threadIdx.x & 31;
    int q = lane >> 3;                 // 0..3 quadrant
    int row = (lane & 7) + ((q & 1) ? 8 : 0);
    int col = (q & 2) ? 8 : 0;
    unsigned a = smem_u32(base + row*stride + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}

// ldmatrix x4 transposed: transposes each 8x8. Register->quadrant mapping (trans):
//   r0=[c0-7,r0-7] r1=[c0-7,r8-15] r2=[c8-15,r0-7] r3=[c8-15,r8-15]
__device__ __forceinline__ void ldm_x4_trans(uint32_t r[4], const bf16* base, int stride){
    int lane = threadIdx.x & 31;
    int q = lane >> 3;
    int row = (lane & 7) + ((q & 1) ? 8 : 0);
    int col = (q & 2) ? 8 : 0;
    unsigned a = smem_u32(base + row*stride + col);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}

// C[16,8] += A[16,16] @ B[16,8]   (fp32 acc, bf16 in)
__device__ __forceinline__ void mma16816(float acc[4], const uint32_t a[4], const uint32_t b[2]){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void cp16(bf16* dst, const bf16* src){
    unsigned s = smem_u32(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src));
}

// Load a [rows, D] tile (D=128) from global g_base at global row0, N=seqlen bound.
__device__ __forceinline__ void load_tile(bf16* smem, const bf16* g_base, int row0, int N, int ROWS){
    const int VECS = ROWS*16;               // D/8 = 16 vecs per row
    for(int vi=threadIdx.x; vi<VECS; vi+=NTHREADS){
        int r = vi >> 4, c8 = (vi & 15)*8;
        bf16* dst = smem + r*D + c8;
        int grow = row0 + r;
        if(grow < N) cp16(dst, g_base + grow*D + c8);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0,0,0,0);
    }
}

__device__ __forceinline__ float warp_row_max4(float v){
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}
__device__ __forceinline__ float warp_row_sum4(float v){
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

__device__ unsigned long long g_prof[5];  // wait, qk, softmax, pv, epilogue (cycles, CTA0)

#ifndef MINCTA
#define MINCTA 2
#endif

// grid = (num_m_tiles, H), block = NTHREADS
extern "C" __global__ void __launch_bounds__(NTHREADS, MINCTA) fa3_sm80_h3_kernel(
        const bf16* __restrict__ Q, const bf16* __restrict__ K,
        const bf16* __restrict__ V, bf16* __restrict__ O,
        int N, float scale)
{
    const int mtile = blockIdx.x;
    const int h     = blockIdx.y;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int g     = lane >> 2;      // 0..7 row group
    const int t     = lane & 3;       // 0..3 in group

    const bf16* Qh = Q + (long)h*N*D;
    const bf16* Kh = K + (long)h*N*D;
    const bf16* Vh = V + (long)h*N*D;
    bf16* Oh       = O + (long)h*N*D;

    extern __shared__ bf16 smem[];
    bf16* sQ = smem;                        // [BM,D] (16KB) reused as sP after Q->regs
    bf16* sK = sQ + BM*D;                    // STAGES x [BN,D]
    bf16* sV = sK + STAGES*BN*D;             // STAGES x [BN,D]
    bf16* sP = sQ;                           // reuse Q region for P (8KB <= 16KB)

    const int row0 = mtile*BM;

    // ---- load Q tile, ldmatrix into registers (resident) ----
    load_tile(sQ, Qh, row0, N, BM);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    uint32_t qf[8][4];               // 8 k-tiles of D, A-frag [16,16]
    #pragma unroll
    for(int kt=0; kt<8; ++kt)
        ldm_x4(qf[kt], sQ + (warp*16)*D + kt*16, D);
    __syncthreads();                 // Q consumed; sQ region now free for sP

    // ---- per-thread running state (rows g and g+8 of this warp) ----
    float acc[16][4];                // O[16,128] : 16 d-tiles
    #pragma unroll
    for(int i=0;i<16;++i){ acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f; }
    float m_lo=-1e30f, m_hi=-1e30f, l_lo=0.f, l_hi=0.f;

    const int NT = (N + BN - 1) / BN;

#if STAGES>1
    // prologue: load tile 0 into buffer 0
    load_tile(sK, Kh, 0, N, BN);
    load_tile(sV, Vh, 0, N, BN);
    asm volatile("cp.async.commit_group;\n");
#endif

    const bool prof = (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0);
    for(int j=0;j<NT;++j){
        long long c0=clock64();
        int cur = j % STAGES;
#if STAGES>1
        if(j+1 < NT){
            int nb = (j+1) % STAGES;
            load_tile(sK + nb*BN*D, Kh, (j+1)*BN, N, BN);
            load_tile(sV + nb*BN*D, Vh, (j+1)*BN, N, BN);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES-1));
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
#else
        load_tile(sK, Kh, j*BN, N, BN);
        load_tile(sV, Vh, j*BN, N, BN);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
#endif
        __syncthreads();
        long long c1=clock64();
        bf16* Kc = sK + cur*BN*D;
        bf16* Vc = sV + cur*BN*D;
        int k0 = j*BN;

        // ---- QK : S[16,64] for this warp ----
        float s[8][4];
        #pragma unroll
        for(int nt=0;nt<8;++nt){ s[nt][0]=s[nt][1]=s[nt][2]=s[nt][3]=0.f; }
        #pragma unroll
        for(int kt=0;kt<8;++kt){
            uint32_t kf[4][4];
            #pragma unroll
            for(int kn=0;kn<4;++kn)        // batch all 4 ldmatrix (overlap latency)
                ldm_x4(kf[kn], Kc + (kn*16)*D + kt*16, D);
            #pragma unroll
            for(int kn=0;kn<4;++kn){
                uint32_t b0[2]={kf[kn][0],kf[kn][2]}; // keys 0-7
                uint32_t b1[2]={kf[kn][1],kf[kn][3]}; // keys 8-15
                mma16816(s[kn*2],   qf[kt], b0);
                mma16816(s[kn*2+1], qf[kt], b1);
            }
        }

        long long c2=clock64();
        // scale + tail mask (key >= N -> -inf)
        #pragma unroll
        for(int nt=0;nt<8;++nt){
            int kbase = k0 + nt*8 + 2*t;
            #pragma unroll
            for(int c=0;c<4;++c){
                int key = kbase + (c&1);
                float val = s[nt][c]*scale;
                s[nt][c] = (key < N) ? val : -1e30f;
            }
        }

        // rowmax over the 64 keys, for row g (lo: c0,c1) and row g+8 (hi: c2,c3)
        float rmax_lo=-1e30f, rmax_hi=-1e30f;
        #pragma unroll
        for(int nt=0;nt<8;++nt){
            rmax_lo = fmaxf(rmax_lo, fmaxf(s[nt][0], s[nt][1]));
            rmax_hi = fmaxf(rmax_hi, fmaxf(s[nt][2], s[nt][3]));
        }
        rmax_lo = warp_row_max4(rmax_lo);
        rmax_hi = warp_row_max4(rmax_hi);

        float nm_lo = fmaxf(m_lo, rmax_lo), nm_hi = fmaxf(m_hi, rmax_hi);
        float al_lo = __expf(m_lo - nm_lo), al_hi = __expf(m_hi - nm_hi);

        // p = exp(s - m_new); rowsum
        float rs_lo=0.f, rs_hi=0.f;
        #pragma unroll
        for(int nt=0;nt<8;++nt){
            s[nt][0]=__expf(s[nt][0]-nm_lo); s[nt][1]=__expf(s[nt][1]-nm_lo);
            s[nt][2]=__expf(s[nt][2]-nm_hi); s[nt][3]=__expf(s[nt][3]-nm_hi);
            rs_lo += s[nt][0]+s[nt][1];
            rs_hi += s[nt][2]+s[nt][3];
        }
        rs_lo = warp_row_sum4(rs_lo);
        rs_hi = warp_row_sum4(rs_hi);

        l_lo = l_lo*al_lo + rs_lo;
        l_hi = l_hi*al_hi + rs_hi;
        m_lo = nm_lo; m_hi = nm_hi;

        // rescale acc
        #pragma unroll
        for(int i=0;i<16;++i){
            acc[i][0]*=al_lo; acc[i][1]*=al_lo;
            acc[i][2]*=al_hi; acc[i][3]*=al_hi;
        }

        long long c3=clock64();

        // ---- PV : O[16,128] += P[16,64] @ V[64,128] ----
        // Build P A-fragments DIRECTLY from the softmax accumulator registers:
        // the mma accumulator layout (m16n8) == A-operand layout (m16k16), same
        // (groupID,tid) mapping, so no smem roundtrip / ldmatrix / shuffle needed.
        uint32_t pf[4][4];
        #pragma unroll
        for(int kt2=0;kt2<4;++kt2){
            __nv_bfloat162 q0=__floats2bfloat162_rn(s[2*kt2][0],   s[2*kt2][1]);
            __nv_bfloat162 q1=__floats2bfloat162_rn(s[2*kt2][2],   s[2*kt2][3]);
            __nv_bfloat162 q2=__floats2bfloat162_rn(s[2*kt2+1][0], s[2*kt2+1][1]);
            __nv_bfloat162 q3=__floats2bfloat162_rn(s[2*kt2+1][2], s[2*kt2+1][3]);
            pf[kt2][0]=*reinterpret_cast<uint32_t*>(&q0);
            pf[kt2][1]=*reinterpret_cast<uint32_t*>(&q1);
            pf[kt2][2]=*reinterpret_cast<uint32_t*>(&q2);
            pf[kt2][3]=*reinterpret_cast<uint32_t*>(&q3);
        }

        #pragma unroll
        for(int kt2=0;kt2<4;++kt2){
            uint32_t vf[8][4];
            #pragma unroll
            for(int dt=0;dt<8;++dt)        // batch all 8 V ldmatrix (overlap latency)
                ldm_x4_trans(vf[dt], Vc + (kt2*16)*D + dt*16, D);
            #pragma unroll
            for(int dt=0;dt<8;++dt){
                uint32_t b0[2]={vf[dt][0],vf[dt][1]}; // d 0-7
                uint32_t b1[2]={vf[dt][2],vf[dt][3]}; // d 8-15
                mma16816(acc[dt*2],   pf[kt2], b0);
                mma16816(acc[dt*2+1], pf[kt2], b1);
            }
        }
        __syncthreads();   // (E) all reads of Kc/Vc/sP done before buffer reuse/overwrite
        if(prof){ long long c4=clock64();
            g_prof[0]+=c1-c0; g_prof[1]+=c2-c1; g_prof[2]+=c3-c2; g_prof[3]+=c4-c3; }
    }

    // ---- epilogue : O = acc / l, store ----
    #pragma unroll
    for(int nt2=0;nt2<16;++nt2){
        int d = nt2*8 + 2*t;
        int gr_lo = row0 + warp*16 + g;
        int gr_hi = row0 + warp*16 + g + 8;
        if(gr_lo < N){
            Oh[(long)gr_lo*D + d + 0] = __float2bfloat16(acc[nt2][0]/l_lo);
            Oh[(long)gr_lo*D + d + 1] = __float2bfloat16(acc[nt2][1]/l_lo);
        }
        if(gr_hi < N){
            Oh[(long)gr_hi*D + d + 0] = __float2bfloat16(acc[nt2][2]/l_hi);
            Oh[(long)gr_hi*D + d + 1] = __float2bfloat16(acc[nt2][3]/l_hi);
        }
    }
}

void fa3_sm80_h3_launch(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O){
    int H = Q.size(1), N = Q.size(2);
    float scale = 1.0f / sqrtf((float)D);
    dim3 grid((N + BM - 1)/BM, H);
    int smem_bytes = (BM*D + 2*STAGES*BN*D) * sizeof(bf16);  // Q/P reuse; K,V x STAGES
    cudaError_t e = cudaFuncSetAttribute(fa3_sm80_h3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    static int printed=0;
    if(!printed){ printed=1; printf("[fa3] smem_bytes=%d setattr=%s\n", smem_bytes, cudaGetErrorString(e)); }
    fa3_sm80_h3_kernel<<<grid, NTHREADS, smem_bytes>>>(
        (const bf16*)Q.data_ptr(), (const bf16*)K.data_ptr(),
        (const bf16*)V.data_ptr(), (bf16*)O.data_ptr(), N, scale);
}

// ---------- micro-test kernels (validate fragment plumbing) ----------
// QK test: S[16,64] = Q[16,128] @ K[64,128]^T for one warp.
extern "C" __global__ void test_qk(const bf16* Q, const bf16* K, float* S){
    const int lane = threadIdx.x & 31;
    const int t = lane & 3, g = lane >> 2;
    extern __shared__ bf16 sm[];
    bf16* sQ = sm; bf16* sK = sm + 16*D;
    for(int i=threadIdx.x;i<16*D;i+=32) sQ[i]=Q[i];
    for(int i=threadIdx.x;i<64*D;i+=32) sK[i]=K[i];
    __syncwarp();
    uint32_t qf[8][4];
    for(int kt=0;kt<8;++kt) ldm_x4(qf[kt], sQ + kt*16, D);
    float s[8][4]; for(int n=0;n<8;++n) s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;
    for(int kt=0;kt<8;++kt) for(int kn=0;kn<4;++kn){
        uint32_t r[4]; ldm_x4(r, sK + (kn*16)*D + kt*16, D);
        uint32_t b0[2]={r[0],r[2]}, b1[2]={r[1],r[3]};
        mma16816(s[kn*2], qf[kt], b0); mma16816(s[kn*2+1], qf[kt], b1);
    }
    for(int nt=0;nt<8;++nt){
        int col=nt*8+2*t;
        S[(g)*64 + col+0]=s[nt][0]; S[(g)*64+col+1]=s[nt][1];
        S[(g+8)*64+col+0]=s[nt][2]; S[(g+8)*64+col+1]=s[nt][3];
    }
}
// PV test: O[16,128] = P[16,64] @ V[64,128] for one warp.
extern "C" __global__ void test_pv(const bf16* P, const bf16* V, float* Oo){
    const int lane = threadIdx.x & 31;
    const int t = lane & 3, g = lane >> 2;
    extern __shared__ bf16 sm[];
    bf16* sP = sm; bf16* sV = sm + 16*BN;
    for(int i=threadIdx.x;i<16*BN;i+=32) sP[i]=P[i];
    for(int i=threadIdx.x;i<64*D;i+=32) sV[i]=V[i];
    __syncwarp();
    uint32_t pf[4][4];
    for(int kt2=0;kt2<4;++kt2) ldm_x4(pf[kt2], sP + kt2*16, BN);
    float acc[16][4]; for(int i=0;i<16;++i) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;
    for(int kt2=0;kt2<4;++kt2) for(int dt=0;dt<8;++dt){
        uint32_t r[4]; ldm_x4_trans(r, sV + (kt2*16)*D + dt*16, D);
        uint32_t b0[2]={r[0],r[1]}, b1[2]={r[2],r[3]};
        mma16816(acc[dt*2], pf[kt2], b0); mma16816(acc[dt*2+1], pf[kt2], b1);
    }
    for(int nt2=0;nt2<16;++nt2){
        int d=nt2*8+2*t;
        Oo[(g)*D + d+0]=acc[nt2][0]; Oo[(g)*D+d+1]=acc[nt2][1];
        Oo[(g+8)*D+d+0]=acc[nt2][2]; Oo[(g+8)*D+d+1]=acc[nt2][3];
    }
}
void test_qk_launch(torch::Tensor Q, torch::Tensor K, torch::Tensor S){
    int sm=(16*D+64*D)*sizeof(bf16);
    cudaFuncSetAttribute(test_qk, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    test_qk<<<1,32,sm>>>((const bf16*)Q.data_ptr(),(const bf16*)K.data_ptr(),S.data_ptr<float>());
}
void test_pv_launch(torch::Tensor P, torch::Tensor V, torch::Tensor O){
    int sm=(16*BN+64*D)*sizeof(bf16);
    cudaFuncSetAttribute(test_pv, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    test_pv<<<1,32,sm>>>((const bf16*)P.data_ptr(),(const bf16*)V.data_ptr(),O.data_ptr<float>());
}

torch::Tensor get_prof(){
    unsigned long long h[5];
    cudaMemcpyFromSymbol(h, g_prof, sizeof(h));
    auto t = torch::empty({5}, torch::kFloat64);
    for(int i=0;i<5;++i) t[i]=(double)h[i];
    return t;
}
void reset_prof(){
    unsigned long long z[5]={0,0,0,0,0};
    cudaMemcpyToSymbol(g_prof, z, sizeof(z));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("fa3", &fa3_sm80_h3_launch, "FA3_SM80_H3");
    m.def("get_prof", &get_prof, "phase cycles");
    m.def("reset_prof", &reset_prof, "reset");
    m.def("test_qk", &test_qk_launch, "test qk");
    m.def("test_pv", &test_pv_launch, "test pv");
}
