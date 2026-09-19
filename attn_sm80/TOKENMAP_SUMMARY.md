# H3 cross-step reuse analysis — empirical trace (verdict: NO cache is useful)

Traced one real video generation (cut2162, turbo) with an env-gated instrumentation node
(custom_nodes/h3_tokenmap): logged, per diffusion step, the packed segment map and whether
each segment's LAYER-0 input/output is identical to the previous step. Raw: tokenmap.jsonl.

## Result
```
generation steps (diffusion) = 6   (turbo LoRA)
transformer layers L = 50
seq_len S = 46052   (this cut; scales with frames/resolution)

segment map (packed [text | ref_img | audio | video]):
  text      1376  ( 3.0%)   conditioning (fixed input)
  ref_img    910  ( 2.0%)   conditioning (fixed input)
  audio      526  ( 1.1%)
  video    43240  (93.9%)   generation target (dynamic)

Q/K/V per layer, EVERY step = [1, 56, 46052, 128]  (non-causal full self-attention)

cross-step change (steps 1..5), fraction of steps a segment changed:
  segment    layer0-INPUT changed   layer0-OUTPUT changed
  text       5/5                    5/5
  ref_img    5/5                    5/5
  audio      5/5                    5/5
  video      5/5                    5/5

per step: static tokens = 0 (0.0%);  dynamic tokens = 46052 (100.0%)
attention calls per video = steps*L = 300
attention FLOPs total = 6*50*4*S^2*D*H ~ 1.82e16
reusable K/V FLOPs ~ 0 ;  non-reusable ~ 1.82e16 (all)
```

## Answers to the 5 questions
1. The full-S attention runs **once per diffusion step per layer** = 6*50 = 300 calls/video.
2. Yes — the same S tokens go through attention every step, but every token's hidden state changes.
3. **None are invariant across steps** (measured). Even the fixed-input conditioning tokens
   (text/ref_img/audio = 6.1%) show changed layer-0 input and definitely changed layer-0 output.
4. text/cond K/V could be cached only at layer 0 (in theory); after layer 0 they are contaminated
   by attending to the changing 93.9% video tokens. Only 6.1% of tokens, one layer of 50. Not worthwhile.
5. ~**0%** of the S^2 attention FLOPs (the expensive 72%) are eliminable: every query attends to all
   keys, and 93.9% of keys (video) change every step, so every query's output changes every step.
   At most the layer-0 K/V *projection* of static tokens (<0.1% of total) — effectively zero.

## Verdict
Do NOT implement prefill / cross-step KV caching for H3. There is no reusable work across
diffusion steps (consistent with the earlier FBCache white-noise failure — zero step redundancy).
The only valid attention lever is making each attention call faster -> FA3_SM80_H3 kernel.
