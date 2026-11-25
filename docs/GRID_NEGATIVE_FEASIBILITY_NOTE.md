# Feasibility of the frozen 3B model on RuleShift-Grid

The 5,500-episode RuleShift-Grid is closed for the frozen
`Qwen/Qwen2.5-VL-3B-Instruct` model at revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. Under the tested strict six-key
full-menu contract, the model did not reliably emit valid probability menus or
demonstrate belief revision in response to public history. It made zero tool
decisions and solved none of the observed base-rule episodes. This assessment
covers the feasibility of the tested model and interface. No confirmatory or
benchmark result is claimed.

## Evidence

The contract requires one probability per candidate and a total within 0.02
of 1.0. The technical gate requires zero parser fallbacks. The earlier map-0 dry
chunk covered five seeds and 11 scenario types, for 55 episodes. All 3,086
decisions had valid menus and zero fallbacks. Across all 2,143 distinct model
requests, the model returned the same step-0 prior, despite 2,125 distinct
public history hashes and 671 distinct image hashes. It made zero tool
decisions and solved 0 of 10 base-rule episodes. See the
[map-0 analysis](GRID_CHUNK0_ANALYSIS.md).

Each two-episode follow-up smoke included one base-rule episode. The
observation repair exposed public walls and adjacent directions. The guard
prevented immediate repeats after recorded `blocked` outcomes, and the updated
semantic gate used those outcomes directly.

| Smoke | Strictly valid menus | Parser fallbacks | Immediate blocked-action repeats | Tool decisions | Base-rule successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| v6 (smoke-13) | 13/120 (10.8%) | 107 | 0/27 | 0 | 0/1 |
| v7 (smoke-14) | 91/120 (75.8%) | 29 | 0/17 | 0 | 0/1 |
| v8 (smoke-15) | 32/120 (26.7%) | 88 | 0/30 | 0 | 0/1 |
| v9 (smoke-16) | 27/120 (22.5%) | 93 | 0/25 | 0 | 0/1 |

In v6, rejected menus omitted candidate IDs. Every cached v7–v9 response had
all six IDs, and each of those three smokes had decisions that failed the sum
check. None of the four smokes passed both gates. See the
[smoke log](GRID_P0_SMOKE_FINDINGS.md).

## Offline normalization audit

For each of v7–v9, the audit took the 119 cached raw responses, ordered each
complete vector by candidate ID, and divided each probability by the vector's
sum. Each cached response counted once, including those the strict parser
rejected. The most common normalized vector still accounted for 68–74% of
responses.

| Prompt | Distinct normalized vectors | Responses with the most common vector |
| --- | ---: | ---: |
| v7 | 7 | 81/119 |
| v8 | 4 | 82/119 |
| v9 | 5 | 88/119 |

Normalization repairs the totals of complete menus. The concentrated vectors
and the unchanged map-0 prior give little evidence of belief revision from
public history. The audit used saved outputs and zero additional GPU time.

Decision counts come from the local `grid_real_decisions.*.jsonl` files,
episode labels and outcomes from the paired `grid_real_evaluator_only.*.jsonl`
files, and the belief audit from `response_cache.*.jsonl`. These files are
under `results/grid_chunks/` for map 0 and
`kaggle_outputs/smoke{13,14,15,16}/results/grid_real/` for the four smokes.

## Cost and disposition

The completed 55-episode T4 chunk took 22,996 seconds (6.39 GPU-hours). At that
rate, 3 maps × 2 seeds × 11 scenario types would take about 7.7 GPU-hours for
66 episodes. The planned 100 maps × 5 seeds × 11 scenario types would take
about 640 GPU-hours before overhead. Both figures are linear projections from
one map.

The current grid stays closed. Reopening requires a changed interface or frozen
model, one two-episode smoke passing both gates, and a resized grid that fits
the remaining quota. Potential method changes include normalized weights,
constrained decoding, a two-stage action and belief readout, or a larger
frozen model. The [closure decision](GRID_FEASIBILITY_DECISION.md) records the
gate and freeze requirements.
