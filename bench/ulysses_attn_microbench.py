"""4-GPU Ulysses sequence-parallel attention microbenchmark on GPU 4,5,6,7.

Measures the achievable EXACT attention speedup vs single-GPU FlashAttention-2 at the
real H3 shape (heads=56, d=128, S~42k), including NCCL all-to-all comm cost on these
A100 80GB PCIe cards (P2P bandwidth is the deciding factor).

Ulysses: seq-sharded [1,H,S/N,d] --all2all--> head-sharded [1,H/N,S,d] --FA2-->
         --all2all--> seq-sharded. Two all-to-all per attention layer.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 ulysses_attn_microbench.py
"""
import os, torch, torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"; dt = torch.bfloat16
    H, d = 56, 128
    assert H % world == 0, "heads must divide world"
    Hl = H // world

    def fa2(q, k, v):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def all2all_seq_to_head(x):
        # x: [1, H, Sl, d] seq-sharded -> [1, Hl, S, d] head-sharded
        _, H_, Sl, d_ = x.shape
        x = x.view(world, Hl, Sl, d_).contiguous()          # split heads into world groups
        y = torch.empty_like(x)
        dist.all_to_all_single(y, x)                          # exchange: group g -> rank g
        # y[g] holds this rank's Hl heads for the tokens that lived on rank g
        return y.permute(1, 0, 2, 3).reshape(1, Hl, world * Sl, d_).contiguous()

    def all2all_head_to_seq(x):
        # x: [1, Hl, S, d] head-sharded -> [1, H, Sl, d] seq-sharded
        _, Hl_, S_, d_ = x.shape; Sl = S_ // world
        x = x[0].view(Hl_, world, Sl, d_).permute(1, 0, 2, 3).reshape(world, Hl_, Sl, d_).contiguous()
        y = torch.empty_like(x)
        dist.all_to_all_single(y, x)
        return y.reshape(1, H, Sl, d_).contiguous()

    def bench(fn, iters=10, warmup=3):
        for _ in range(warmup): fn()
        torch.cuda.synchronize(); dist.barrier()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters): fn()
        en.record(); torch.cuda.synchronize()
        t = st.elapsed_time(en) / iters
        tt = torch.tensor([t], device=dev); dist.all_reduce(tt, op=dist.ReduceOp.MAX)
        return tt.item()

    # ---- correctness: Ulysses output must equal single-GPU full attention ----
    Sc = 4096; Slc = Sc // world
    torch.manual_seed(0)
    qf = torch.randn(1, H, Sc, d, device=dev, dtype=dt); kf = torch.randn_like(qf); vf = torch.randn_like(qf)
    ref = fa2(qf, kf, vf)                                    # [1,H,Sc,d] full
    q_sh = qf[:, :, rank*Slc:(rank+1)*Slc, :].contiguous()  # this rank's seq shard
    k_sh = kf[:, :, rank*Slc:(rank+1)*Slc, :].contiguous()
    v_sh = vf[:, :, rank*Slc:(rank+1)*Slc, :].contiguous()
    o_sh = all2all_head_to_seq(fa2(all2all_seq_to_head(q_sh), all2all_seq_to_head(k_sh), all2all_seq_to_head(v_sh)))
    ref_sh = ref[:, :, rank*Slc:(rank+1)*Slc, :]
    err = (o_sh.float() - ref_sh.float()).abs()
    emax = err.max(); emean = err.mean()
    dist.all_reduce(emax, op=dist.ReduceOp.MAX); dist.all_reduce(emean, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[correctness] Ulysses vs single-GPU FA2: max_err={emax.item():.4e} mean_err={emean.item()/world:.4e}", flush=True)

    for S in (40960, 49152):
        Sl = S // world
        q = torch.randn(1, H, Sl, d, device=dev, dtype=dt)
        k = torch.randn_like(q); v = torch.randn_like(q)
        # full Ulysses layer: 2 all-to-all + FA2 on Hl heads over full S
        def ulysses():
            qh = all2all_seq_to_head(q); kh = all2all_seq_to_head(k); vh = all2all_seq_to_head(v)
            oh = fa2(qh, kh, vh)
            return all2all_head_to_seq(oh)
        # comm-only (measure all-to-all overhead)
        def commonly():
            return all2all_seq_to_head(q)
        # single-GPU equivalent (full S, all H) for reference on rank 0 only
        t_uly = bench(ulysses)
        t_comm = bench(commonly)
        if rank == 0:
            qf = torch.randn(1, H, S, d, device=dev, dtype=dt); kf = torch.randn_like(qf); vf = torch.randn_like(qf)
            for _ in range(3): fa2(qf, kf, vf)
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True); st.record()
            for _ in range(10): fa2(qf, kf, vf)
            en.record(); torch.cuda.synchronize(); t_single = st.elapsed_time(en) / 10
            print(f"S={S}: single-GPU FA2={t_single:.2f}ms/layer | 4-GPU Ulysses={t_uly:.2f}ms/layer "
                  f"(comm~{t_comm:.2f}ms×3q) | speedup={t_single/t_uly:.2f}x | est 50-layer attn: "
                  f"{t_single*50/1000:.2f}s -> {t_uly*50/1000:.2f}s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
