# SYNTHETIC CPU SMOKE — NOT A SCIENTIFIC FINDING

Synthetic CPU-only smoke run of the RuleShift-Grid plumbing. These numbers are **not** a benchmark result, a bound, or a scientific finding.

- Config: `configs/grid_dev.yaml`
- Quick mode: True
- Maps: 2; seeds: 1; scenarios: 22
- Methods: no_adaptation, public_advisor, random_probe, info_gain, oracle

## Per-method summary

### no_adaptation

| Metric | Value |
| --- | --- |
| success_rate | 0.6818 |
| mean_return | -1.1295 |
| mean_total_cost | 22.5909 |
| false_alarm_rate | 0.0000 |
| detection_rate | 0.0000 |

### public_advisor

| Metric | Value |
| --- | --- |
| success_rate | 0.6818 |
| mean_return | -1.1295 |
| mean_total_cost | 22.5909 |
| false_alarm_rate | 0.0000 |
| detection_rate | 0.0000 |

### random_probe

| Metric | Value |
| --- | --- |
| success_rate | 0.6818 |
| mean_return | -1.1295 |
| mean_total_cost | 22.5909 |
| false_alarm_rate | 0.0000 |
| detection_rate | 0.0000 |

### info_gain

| Metric | Value |
| --- | --- |
| success_rate | 0.5000 |
| mean_return | 1.6341 |
| mean_total_cost | 41.8636 |
| false_alarm_rate | 0.0000 |
| detection_rate | 0.0000 |
| mean_adaptation_steps | 4.0000 |

### oracle

| Metric | Value |
| --- | --- |
| success_rate | 1.0000 |
| mean_return | 9.5864 |
| mean_total_cost | 4.6364 |
| false_alarm_rate | 0.0000 |
| detection_rate | 0.0000 |

## Paired cost vs no_adaptation (scenario-level bootstrap)

| Method | Mean difference | Low | High | Scenarios |
| --- | --- | --- | --- | --- |
| public_advisor | +0.0000 | +0.0000 | +0.0000 | 22 |
| random_probe | +0.0000 | +0.0000 | +0.0000 | 22 |
| info_gain | +19.2727 | +7.2273 | +30.6364 | 22 |
| oracle | -17.9545 | -28.3636 | -7.6818 | 22 |

Negative differences mean the method spent fewer total cost units. The scenario (not the seed or frame) is the resampling unit.
