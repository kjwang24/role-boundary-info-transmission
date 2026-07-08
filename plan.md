# Plan: `roles_activation_patching` — think→answer information transmission in DeepSeek-R1-Distill-Qwen-1.5B

## Context

**Research question.** *Is answer-relevant information lost / not transmitted across the boundary
between the end of the `<think>` block and the start of the assistant answer?* Concretely, we focus on
two senses of "loss":
- **Sense 2 — representational:** is the answer-relevant information *decodable* from the residual
  stream at the end of `<think>`, and does that decodability *survive* to the first answer position?
- **Sense 3 — routing:** does the answer position actually *read* the think/boundary information via
  attention, or is there a bottleneck at the handoff?

(The purely functional "does the CoT matter at all" sense is out of scope by choice.)

**Model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — a genuine reasoning model that natively emits
`<think> … </think>` blocks, **MIT-licensed / ungated**, ~1.78B params. `<think>`/`</think>` are
ordinary text tokens; the chat template auto-injects a leading `<｜Assistant｜><think>\n`.

## Methods — how each answers the question (lean, two techniques)

- **Sense 2 → logit lens + linear probes.** Measure decodability of the answer variable in the
  residual stream at each layer, read at two loci: **end-of-think (`</think>`)** and the **first answer
  position**. The *drop* in decodability between them is the representational-transmission measure.
- **Sense 3 → attention knockout.** Ablate the answer position's attention edges to the
  think/boundary positions (one layer at a time, and grouped interior-content vs boundary) and measure
  answer degradation (KL + logit-diff). Tests whether the answer causally reads across the handoff.

*(We deliberately drop activation patching and attribution patching — attention knockout is the sharper
causal-routing test for this question, and keeping the method set to two avoids bloat.)*

**Synthesis — the 2×2 grid.** Probe/logit-lens result (rows) × attention-knockout result (columns);
both techniques form the axes, so every item is placed by a joint reading:

| | Knockout hurts (answer reads think) | Knockout no effect (answer ignores think) |
|---|---|---|
| **Decodable at handoff** | **Transmitted & used** — present and read; no loss. | **Lost at the handoff (sense 3)** — info sits at `</think>` but is never read. *Headline loss case.* |
| **Not decodable at handoff** | **Nonlinear / late-consolidated** — read exists but the variable isn't *linearly* at the boundary. | **Never consolidated** — nothing there, nothing read. |

Plus the standalone **end-of-think → answer decodability drop** as the direct sense-2 readout.

## Platform / environment

- **Runs on Google Colab, T4 GPU (16 GB).** T4 is more than enough for a 1.78B model.
- **Precision: `float16`.** Turing (T4) has fp16 tensor cores but **no bf16** — so fp16 (not bf16).
  Qwen2 has no softcapping and is generally fp16-stable; fp32 (~7 GB) also fits if instability appears.
- Memory is a non-issue on T4 (fp16 weights ≈ 3.5 GB), and runs complete in minutes — the CPU/RAM/OOM
  constraints of the earlier local plan no longer apply.
- **Model facts (config.json):** `model_type=qwen2`, `n_layers=28`, `d_model=1536`, `n_heads=12`,
  `n_kv_heads=2` (GQA), `head_dim=128`, `d_mlp=8960` (SiLU), `vocab=151936`, `n_ctx=131072`, RoPE
  θ=10000, RMSNorm, no softcapping. Embeddings **untied on disk** (`tie_word_embeddings=false`).
- Ungated → no HF token required (a token only helps download rate limits).

## Deliverable

A single self-contained **Colab notebook**: `roles_boundary_transmission.ipynb` (no `.py` mirror).
It installs deps, runs the whole experiment on the T4, renders all plots inline, and writes a short
analysis. Figures + a `metrics.json` + `analysis.txt` are also saved to the working dir for download.

## Notebook sections

**1. Setup / install** — `pip install transformer_lens transformers matplotlib scikit-learn` (torch is
preinstalled on Colab); imports; **device/dtype auto-detect** (`cuda`→`float16`, else CPU→`float32`);
seed. Brief markdown stating the research question + 2×2 frame.

**2. Model loading** — DeepSeek is **not in the TL registry**, so use the `hf_model` + Qwen-alias pattern:
```python
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
tok      = AutoTokenizer.from_pretrained(MODEL_NAME)
model    = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", hf_model=hf_model, tokenizer=tok,
    device=device, dtype="float16", fold_ln=False,
    center_writing_weights=False, center_unembed=False)
```
- **Config-parity assert** before building: `hf_model.config` vs the resolved TL config must match on
  `n_layers=28`, `d_model=1536`, `n_heads=12`, `n_kv_heads=2`, `d_mlp=8960`, `vocab=151936` — fail loudly
  on drift so DeepSeek weights can never be poured into the wrong skeleton.
- Optional one-shot **HF-vs-TL logit validation** on a clean prompt (same argmax / small max-abs diff).

**3. Dataset + clean/corrupt forward passes** — `build_dataset()`:
- **One fixed template with a varying answer fact**, generating **~100–200 token-aligned examples** (a
  size probes need; trivial on T4). Scaffold = DeepSeek chat + opened `<think>` … reasoning that states
  the answer fact (which appears **only** inside the think block) … `</think>` + a teacher-forced
  `ANSWER_PREFIX` + the answer token. Fixed structure keeps positions aligned across all examples.
- Record **role-labeled positions**: interior think content, the **boundary** (last think token,
  `</think>`), and the **first answer position(s)**.
- **Minimal-pair** partner per example (same reasoning, only the answer fact swapped, e.g. 7→3) — the
  causal contrast used by logit-lens deltas and attention knockout.
- Run clean (and corrupt) with `run_with_cache`, caching per-layer residuals (`resid_post`) at the
  labeled positions + attention patterns for the knockout step. Metric anchors: KL(clean‖·) (primary)
  and logit-diff between the clean answer token and the minimal-pair's answer token (secondary), scored
  at the answer content position (not a formatting token).
- **Assert** alignment (`len` equal, identical outside the think span) and that the corruption actually
  changes the answer.

**4. Logit lens (sense 2, training-free)** — for each layer × labeled position, apply `ln_final` then
the unembed to the cached residual and read the answer token's rank/logit/prob. Produce the
**decodability trajectory** across positions and layers, highlighting **end-of-think vs answer
position**. Cheap: reuses the cache, no extra forward passes.

**5. Linear probes (sense 2, quantitative)** — train cross-validated logistic probes on the cached
residuals to decode the answer variable at each layer, at **end-of-think** and the **answer position**.
Report CV accuracy per layer/locus and the **end-of-think → answer accuracy drop** (the representational
transmission/loss number).

**6. Attention knockout (sense 3)** — hook `blocks.L.attn.hook_pattern` (or `hook_attn_scores`) and zero
the answer-position query's attention to a target key group (interior think content vs boundary),
one layer at a time. Measure answer degradation (KL + logit-diff) → **knockout-effect-by-layer**,
separated by target region. This is the causal read-through / bottleneck test.

**7. Synthesis + visualization** — (consult the `dataviz` skill first) render: the decodability
trajectory plot, the probe accuracy-by-layer plot with the boundary drop annotated, the
knockout-effect-by-layer plot, and the **2×2 placement** (per-item and aggregate). Save PNGs +
`metrics.json`.

**8. Analysis** — answer the question explicitly in-notebook and in `analysis.txt`: is the info present
at `</think>` (logit lens/probe)? Does it survive to the answer position (decodability drop)? Does the
answer causally read it (knockout)? → which 2×2 cell dominates, and therefore whether information is
lost/not transmitted across the boundary, and at which layers the handoff resolves.

## Verification (end-to-end)

1. Notebook runs top-to-bottom on a Colab T4 without OOM; model loads in fp16; config-parity assert passes.
2. HF-vs-TL logit check agrees on the clean prompt.
3. Sequence sanity: aligned lengths, identical outside the think span, minimal-pair changes the answer.
4. Method sanity: logit lens shows the answer token becoming decodable in the think block; probe CV
   accuracy is well above chance at some layer; knockout of *some* region measurably degrades the answer
   (else the metric/positions are mis-set).
5. The three readouts populate the 2×2 and the written analysis states the conclusion with the
   supporting numbers (decodability drop, knockout Δ, dominant cell).

## Notes / decisions
- Kept from earlier design: minimal-pair aligned sequences, role-labeled boundary/answer positions,
  KL + logit-diff metrics, config-parity assert, `hf_model` + Qwen-alias loading recipe.
- Dropped: local CPU run, activation-patching residual sweep, attribution patching, `.py` mirror,
  functional/behavioral faithfulness sense.
- No file is written outside `roles_activation_patching/`.
