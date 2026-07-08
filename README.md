# roles_activation_patching — think→answer information transmission

Does a reasoning model **lose information across the boundary** between the end of its `<think>`
block and the start of its answer? This project probes that on
[`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
with TransformerLens, focusing on two senses of "loss":

- **Representational** — is the answer *decodable* at the end of `<think>`, and does that survive to
  the first answer position? → **logit lens + linear probes**
- **Routing** — does the answer position causally *read* the think/boundary info via attention? →
  **attention knockout**

Results are combined into a 2×2 verdict (decodable-at-handoff × knockout-hurts).

## How to run (Google Colab, T4)

1. Open `roles_boundary_transmission.ipynb` in Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** The first cell installs `transformer_lens` + `scikit-learn` (torch and
   matplotlib are preinstalled on Colab).

The model is MIT-licensed and **ungated**, so no HuggingFace token is required. The full run takes a
few minutes on a T4; it loads in **float16** (Turing has fp16 tensor cores but no bf16). ~1.78 B params
fits comfortably in 16 GB.

## Outputs (written to `outputs/`)

- `summary.png` — 3 panels: logit-lens decodability, probe accuracy, and attention-knockout effect,
  all by layer.
- `metrics.json` — per-layer arrays + headline scalars + the 2×2 verdict.
- `analysis.txt` — the written interpretation (also printed in the notebook).

## Method notes

- The task plants a **secret value only inside `<think>`**, so the answer must cross the boundary to be
  produced (nothing leaks from the question). Sequences use a fixed template with single-token
  item/value slots so all examples are token-aligned; a **minimal pair** swaps only the value.
- DeepSeek is not in the TransformerLens registry, so its weights are loaded into the
  `Qwen/Qwen2.5-1.5B-Instruct` skeleton (identical architecture) via `hf_model=...`, guarded by a
  config-parity assert.
- Caveat: single-layer attention knockout is a direct-edge test; multi-hop routing
  (think → intermediate positions → answer) is only fully severed by the block-all-layers result.

## Local / non-Colab

The notebook auto-detects device (`cuda`→fp16, else CPU→fp32). CPU works but is slow for a 1.78 B model.
See `requirements.txt` for dependencies (`plan.md` documents the earlier CPU-constrained design).
