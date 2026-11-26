# Week 9 - Main held-out evaluation (SYNTHETIC)

**SYNTHETIC.** This dry run used the deterministic stub and a development fixture. It provides no empirical VLM result.

Seeds are averaged within each scenario. Intervals resample scenarios. All model methods had a cap of 6 calls per episode.

## Paired differences (proposed minus baseline)

| Comparison | Scenarios | Mean difference | 95% paired CI |
|---|---:|---:|---|
| m1_stub − direct_stub | 4 | -2.500 | [-10.000, +5.000] |
| m1_stub − diva_stub | 4 | +7.500 | [+2.500, +10.000] |
| m1_stub − immediate_stub | 4 | +0.000 | [+0.000, +0.000] |

## Cost and call accounting

| Method | Episodes | Mean return | Probes | VLM calls | Tokens | Invalid actions | Cap fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_stub | 4 | +7.000 | 11 | 24 | 5404 | 0 | 114 |
| diva_stub | 4 | -3.000 | 24 | 24 | 14164 | 0 | 24 |
| m1_stub | 4 | +4.500 | 0 | 24 | 14164 | 0 | 192 |
| immediate_stub | 4 | +4.500 | 0 | 24 | 14164 | 0 | 192 |

## Negative and inconclusive results

All comparisons whose paired interval does not support the proposed method are reported here rather than omitted:

- m1_stub minus direct_stub: interval includes zero.
- m1_stub minus immediate_stub: interval entirely at or below zero.