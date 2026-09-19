"""MiniMax-H3 First-Block-Cache (TeaCache-style step caching) — env-gated.

Activates ONLY when H3_FBCACHE=1, so it is inert for normal fleet runs.

Idea (approximate, quality-tunable): across denoise steps the DiT's block-stack
transformation changes slowly. Each step we run ONLY block 0 and measure how much
its residual moved vs the previous step (relative L1). While the *accumulated* motion
stays under a threshold we SKIP the remaining 49 blocks and reuse the cached
"rest-of-stack" residual. When it exceeds the threshold we recompute all blocks and
refresh the cache. Step 0 of every generation always computes in full.

This multiplies throughput on EVERY GPU (unlike sequence-parallel, it does not consume
extra GPUs), so it stacks with the data-parallel fleet.

KNOWN ISSUE (see benchmark_log.md, 2026-09-19): the rel-L1 signal is a MEAN over the
whole packed sequence [text|cond|audio|video]. The static conditioning tokens dominate
that mean, so the signal reads ~0 and the cache skips EVERY step after step 0 (measured
accum=0.000, 5/6 turbo steps skipped) -> over-aggressive, quality collapses toward a
1-step generation. FIX (TODO): restrict the change signal to the video/audio (denoised)
tokens only, or switch to a t_emb-based TeaCache signal. Do not deploy as-is.

Env:
  H3_FBCACHE=1            enable
  H3_FBCACHE_THRESH=0.08  accumulated rel-L1 skip threshold (higher = more skips, more drift)
  H3_FBCACHE_WARMUP=1     always fully compute the first N steps of each gen (default 1)
  H3_FBCACHE_MAXSKIP=0    cap consecutive skips (0 = uncapped)
  H3_FBCACHE_LOG=1        print per-step skip decisions + per-generation stats

Mutually exclusive with the sequence-parallel run_blocks patch (don't enable both).
"""
import os
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if os.environ.get("H3_FBCACHE") == "1":
    import torch
    import comfy.ldm.minimax.model as M

    THRESH = float(os.environ.get("H3_FBCACHE_THRESH", "0.08"))
    WARMUP = int(os.environ.get("H3_FBCACHE_WARMUP", "1"))
    MAXSKIP = int(os.environ.get("H3_FBCACHE_MAXSKIP", "0"))
    LOG = os.environ.get("H3_FBCACHE_LOG") == "1"

    _c = {
        "prev_first": None,   # block-0 residual from last step (the skip signal)
        "prev_rest": None,    # cached residual of blocks[1:] (what we reuse on a skip)
        "accum": 0.0,
        "last_ts": None,
        "step": 0,            # step index within the current generation
        "run_skips": 0,       # consecutive skips
        # stats
        "g_steps": 0, "g_skips": 0,
    }

    def _reset():
        _c["prev_first"] = None; _c["prev_rest"] = None
        _c["accum"] = 0.0; _c["step"] = 0; _c["run_skips"] = 0
        _c["g_steps"] = 0; _c["g_skips"] = 0

    def _rel_l1(a, b):
        return (a - b).abs().mean() / (b.abs().mean() + 1e-8)

    _orig_run_blocks = M.MiniMaxH3Model.run_blocks

    def _fbcache_run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options={}):
        blocks = self.blocks
        original = h
        # always run block 0
        h = blocks[0](h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        first_residual = h - original

        should_calc = True
        if (_c["step"] >= WARMUP and _c["prev_first"] is not None and _c["prev_rest"] is not None
                and _c["prev_first"].shape == first_residual.shape
                and _c["prev_rest"].shape == h.shape):
            diff = _rel_l1(first_residual, _c["prev_first"]).item()
            _c["accum"] += diff
            capped = MAXSKIP > 0 and _c["run_skips"] >= MAXSKIP
            if _c["accum"] < THRESH and not capped:
                should_calc = False
            else:
                _c["accum"] = 0.0

        if should_calc:
            for blk in blocks[1:]:
                h = blk(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
            _c["prev_rest"] = (h - (original + first_residual)).detach()  # residual of blocks[1:]
            _c["run_skips"] = 0
        else:
            h = h + _c["prev_rest"]  # reuse blocks[1:] transformation
            _c["run_skips"] += 1
            _c["g_skips"] += 1

        if LOG:
            _d = _c["accum"]
            print(f"[h3_fbcache] step {_c['step']} {'SKIP' if not should_calc else 'calc'} accum={_d:.3f}", flush=True)

        _c["prev_first"] = first_residual.detach()
        _c["step"] += 1
        _c["g_steps"] += 1
        return h

    _orig_forward = M.MiniMaxH3Model._forward

    def _fbcache_forward(self, x, timestep, *a, **k):
        try:
            ts = float(timestep.flatten()[0].item())
        except Exception:
            ts = None
        # sigmas decrease within a generation; a jump up (or first call) = new gen -> reset
        if ts is not None and (_c["last_ts"] is None or ts > _c["last_ts"] + 1e-6):
            if LOG and _c["g_steps"] > 0:
                print(f"[h3_fbcache] gen done: {_c['g_steps']} steps, {_c['g_skips']} skipped "
                      f"({100*_c['g_skips']/max(_c['g_steps'],1):.0f}%) thresh={THRESH}", flush=True)
            _reset()
        _c["last_ts"] = ts
        return _orig_forward(self, x, timestep, *a, **k)

    M.MiniMaxH3Model.run_blocks = _fbcache_run_blocks
    M.MiniMaxH3Model._forward = _fbcache_forward
    print(f"[h3_fbcache] ENABLED thresh={THRESH} warmup={WARMUP} maxskip={MAXSKIP or 'off'}", flush=True)
