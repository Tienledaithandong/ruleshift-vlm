# VLM Prompt Anchoring × Belief Conditioning Diagnostic — Analysis

**Experiment:** `vlm-prompt-belief-diagnostic-v1`  
**Evidence:** 48/48 real requests completed; 48/48 valid; all four arms 12/12 `COMMIT`  
**Status:** exploratory development diagnostic, not confirmatory  

---

## 1. The question the design could answer

The design crossed two factors, each at two levels, on the same 12 development contexts:

| Factor | Level 0 | Level 1 |
|---|---|---|
| Belief text | prior/evidence only | explicit posterior |
| Output instruction | original anchored action example | neutral schema-only instruction |

This yields four arms: B1, B2, B3, B4. The matched pairs isolate:

| Comparison | Isolates |
|---|---|
| B2 − B1 | output-format anchoring, without explicit posterior |
| B4 − B3 | output-format anchoring, with explicit posterior |
| B3 − B1 | explicit posterior, under the anchored format |
| B4 − B2 | explicit posterior, under the neutral format |
| `(B4 − B3) − (B2 − B1)` | descriptive interaction |

The original A1 was intentionally not reused as a replacement for B1. B1 is a matched replication: exact original A1 prompt, same 12 contexts, same parser and model. The original 12 A1 responses remain historical comparison only.

---

## 2. Headline result

**All four arms produced exactly the same action in every context: `COMMIT`.** There were 48 valid actions and zero fallbacks. The four arms are therefore numerically identical on every decision metric:

- optimal accuracy `1/3 = 0.3333333333333333`,
- mean regret `2.0928107724292278`,
- max regret `5.025`,
- over-probe `0.0`,
- under-probe `2/3 = 0.6666666666666666`,
- one distinct action.

The result is not a small improvement that misses a threshold. It is a complete collapse to one action in all four conditions, including the condition that supplies an explicit posterior and the condition that removes the anchored action example.

---

## 3. Hypothesis-by-hypothesis interpretation

### 3.1 Prompt-format anchoring

**Evidence:** B2 − B1 = 0 action changes; B4 − B3 = 0 action changes. Accuracy and mean-regret deltas are zero in both pairs.

**Interpretation:** The anchored output example is not necessary to produce the observed `COMMIT` behavior. Replacing it with a neutral schema-only instruction changes the serialization (pretty-printed JSON, four extra output tokens) but not the chosen action. On this sample, prompt-format anchoring is **not a sufficient explanation** for the failure.

**Limit:** This does not show that anchoring can never matter, nor that a different neutral phrasing would be equivalent. It shows that this exact neutral footer, with all other prompt content matched, did not move the model off `COMMIT`.

### 3.2 Belief computation / belief conditioning

**Evidence:** B3 − B1 = 0 action changes; B4 − B2 = 0 action changes. Supplying an explicit public posterior, including the posterior probabilities/values already used by the reference, produced no behavioral change.

**Interpretation:** The model's action output is insensitive to the supplied posterior text in this interface. This is **not** evidence that the model computes the posterior internally. It is evidence that giving it the posterior in text did not rescue the decision. The failure cannot be attributed solely to “the model was not told the posterior.”

**Limit:** The experiment did not probe internal representations, attention, or intermediate beliefs. A model could ignore the posterior text, encode it but fail to use it, or use it and still prefer `COMMIT`. The design cannot separate those mechanisms.

### 3.3 Probe-versus-commit decision limitation

**Evidence:** Under every prompt and belief condition, the model selected `COMMIT` 48/48 times, including 8 contexts where the reference-optimal action was a probe and the regret of committing was as high as `5.025`.

**Interpretation:** The data are **consistent with** a decision-stage commit bias or a failure to map available information into the probe actions. This is the remaining locus after the two prompt-side manipulations were neutralized. But the experiment does not isolate the decision stage: the same output is also consistent with the model ignoring the posterior text entirely.

**What can be said:** The failure is not repaired by removing the anchored example or by supplying an explicit posterior. It survives both. That makes a pure prompt-format explanation and a pure “missing posterior” explanation insufficient.

**What cannot be said:** “The model computes correct beliefs but chooses commit” is not established. The posterior was supplied, not verified as internally represented. The correct formulation is “the action policy is constant-`COMMIT` even when the posterior is explicit.”

### 3.4 Interaction

**Evidence:** `(B4 − B3) − (B2 − B1) = 0.0` for accuracy and mean regret; zero action changes.

**Interpretation:** On this sample there is no detectable interaction between output format and posterior text. A degenerate 2×2 with all cells at the same action cannot support a claim of a general null interaction; it supports only “no interaction was observed here.”

---

## 4. Comparison with the historical A1 and CPU baselines

| Condition | Accuracy | Mean regret | Relation |
|---|---:|---:|---|
| Historical A1 (12 calls) | 0.3333333333333333 | 2.0928107724292278 | original feasibility failure |
| B1 (matched replication) | 0.3333333333333333 | 2.0928107724292278 | exactly the same output pattern |
| B2 | 0.3333333333333333 | 2.0928107724292278 | no rescue |
| B3 | 0.3333333333333333 | 2.0928107724292278 | no rescue |
| B4 | 0.3333333333333333 | 2.0928107724292278 | no rescue |
| `always_commit` | 0.3333333333333333 | 2.0928107724292278 | exactly equal |
| `information_gain` | 0.9166666666666666 | 0.08333333333333341 | much better |
| `exact` | 1.0 | 0.0 | reference |

B1's total input tokens are exactly the original A1 total, **8,049**. Its 12 outputs are the same compact `COMMIT` JSON. This is a strong matched replication of the historical A1 outcome, not a new independent sample.

The arms are exactly `always_commit` on the decision metrics. They are worse than the CPU `information_gain` baseline by `0.5833333333333333` accuracy and `2.0094774390958943` mean regret. The model is not merely below the visual-smoke gate; it is at the commit-only floor.

---

## 5. Why the 2×2 is degenerate

A 2×2 design is informative when at least one factor changes the response. Here every cell has the same response vector: 12 `COMMIT`. Consequently:

1. All paired action-change counts are zero.
2. All paired accuracy/regret deltas are zero.
3. The interaction estimate is zero.
4. The class-level accuracy pattern is identical: `COMMIT` 4/4, `PROBE_SIGN` 0/4, `PROBE_BLUE` 0/4.
5. The only observed difference is serialization: anchored prompts emit compact JSON with 8 output tokens; neutral prompts emit pretty-printed JSON with 12 output tokens.

The design therefore does **not** identify a causal path inside the model. It identifies a boundary: two prompt-side interventions that might have rescued the behavior did not.

---

## 6. Output-format effect without action change

The neutral footer specified the schema without concrete action examples. The model complied with the schema in both formats:

| Arm | Format | Raw output shape | Output tokens | Parse |
|---|---|---|---:|---|
| B1 | anchored | compact JSON | 8 | valid |
| B2 | neutral | pretty-printed JSON | 12 | valid |
| B3 | anchored | compact JSON | 8 | valid |
| B4 | neutral | pretty-printed JSON | 12 | valid |

This is a useful control: the neutral instruction worked as an instruction (it changed formatting) but not as a decision intervention. The model was not failing because it could not understand the neutral schema; it produced a valid neutral JSON object every time.

---

## 7. Token, latency, and memory observations

| Arm | Input tokens total | Output tokens total | Latency mean | GPU time |
|---|---:|---:|---:|---:|
| B1 | 8,049 | 96 | 2,239.868 ms | 26.801 s |
| B2 | 8,073 | 144 | 2,288.864 ms | 27.425 s |
| B3 | 8,852 | 96 | 2,296.101 ms | 27.507 s |
| B4 | 8,876 | 144 | 2,490.140 ms | 29.839 s |
| All | 33,850 | 480 | 2,328.743 ms | 111.571 s |

Explicit posterior prompts add roughly 800–830 input tokens. Neutral prompts add four output tokens because of pretty printing. Latency differences are small and descriptive; the balanced order reduces, but does not eliminate, warm-up and contention effects.

Peak generation memory was `7,840,860,160` bytes allocated and `8,097,103,872` bytes reserved on a 15,636,037,632-byte T4. Post-load peak was `7,628,753,920` bytes. These are descriptive runtime measurements, not scientific endpoints.

---

## 8. Precision audit

The run used `Qwen/Qwen2.5-VL-3B-Instruct` at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.

| Property | Recorded value |
|---|---|
| Parameter count | 3,754,622,976 |
| Parameter dtype | `torch.bfloat16` (824 tensors) |
| Model config dtype | `torch.bfloat16` |
| Attention | `sdpa` |
| Autocast enabled | false |
| Autocast configured dtype | `torch.float16` |
| Explicit precision switch | false |
| GPU | Tesla T4, capability 7.5 |
| Native BF16 | **false** |
| `is_bf16_supported()` flag | true (not native evidence) |

T4 has no native BF16 tensor cores. The audit does **not** claim native BF16 execution merely because the run succeeded. Parameters remained BF16, no autocast was enabled, and no precision condition was silently changed. The run therefore used BF16 parameters on a GPU where BF16 arithmetic is not hardware-native; this is a disclosed runtime condition, not a new condition introduced after the freeze.

---

## 9. Statistical and inferential limits

- **Development subset:** 12 contexts, previously exposed; this is not a holdout or confirmatory sample.
- **One model / one revision / one decoding configuration / one GPU session.**
- **Constant outcome:** zero variance across contexts and arms; paired differences are exactly zero. No interval or significance test would be meaningful.
- **No image input:** this is a text-only diagnostic; it does not test visual grounding.
- **No internal access:** the experiment manipulates public text and observes actions; it does not measure beliefs.
- **Tool-selection accuracy:** undefined because no probe occurred; reporting it as 0 would be wrong.
- **Two failed infrastructure attempts:** v1 and v2 made zero model calls; the only real evidence is v3.
- **Runtime:** T4 BF16 is non-native; latency/memory may not transfer to other hardware.

---

## 10. Conclusion

The diagnostic answers its narrow question negatively for this 3B model:

1. **Prompt-format anchoring is not the sole cause.** Neutralizing the anchored action example changes serialization but not action.
2. **Absence of an explicit posterior is not the sole cause.** Supplying the posterior changes input tokens but not action.
3. **The observed behavior is a constant `COMMIT` action policy.** The design cannot decide whether the posterior is ignored, misrepresented, or overridden at decision time. It can say the policy is insensitive to both interventions.

This is a stronger negative result than the original A1 alone because it includes matched format and belief controls, all of which are null. It is still a small, exposed development sample and must not be presented as confirmation.

**The 3B feasibility line should be closed.** The next decision is whether to authorize a separately preregistered feasibility experiment with a stronger model. That decision is outside this experiment and is recorded in `docs/VLM_PROMPT_BELIEF_NEXT_GATE.md`.
