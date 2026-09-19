# The `run_blocks` seam

To let `harness/sp_runtime.py` override the 50-block loop cleanly (and to keep the
override *bit-exact*), extract the block loop from `MiniMaxH3Model._forward` into a
dedicated method. This is **behavior-identical** — normal single-GPU generation runs
exactly as before; the SP runtime simply monkeypatches this one method.

File: `comfy/ldm/minimax/model.py`, class `MiniMaxH3Model`.

## 1. Add the method

```python
def run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options={}):
    # Default single-GPU block loop. Overridden (monkeypatched) by the SP runner
    # to execute the 50 blocks with Ulysses sequence parallelism across GPUs.
    device = h.device
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return h
```

## 2. Call it from `_forward`

Replace the inline block loop in `_forward` with a single call:

```python
h = self.run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
```

That's the whole seam. `sp_runtime.install()` then does:

```python
MM.MiniMaxH3Model.run_blocks = _sp_run_blocks   # shard -> 50 blocks Ulysses -> all_gather
MM.Attention.forward         = _sp_attn_forward # all-to-all around FlashAttention-2
```

Verified: with `WORLD=1` the patched `run_blocks` falls through to the original, and
with `WORLD=4` the seam replay is `max_err 0.0` vs the single-GPU baseline.
