# Frozen-VLM offline pilot (empirical run)

Model: `Qwen/Qwen2.5-VL-3B-Instruct`
Device: `cuda`
Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`

This report is produced only when the real backend is used. It still requires preregistration and a frozen test configuration before any hypothesis claim is made.

| Diagnostic | Value |
|---|---:|
| Contexts | 3 |
| Model calls | 3 |
| JSON valid rate | 0.000 |
| Valid action rate | 0.000 |
| Over-probe rate | 0.000 |
| Under-probe rate | 0.667 |
| Mean decision regret | 10.000 |
| Mean output tokens | 522.0 |
| Cache hit rate | 0.000 |

Public decisions and privileged labels are written to separate files and must never be joined into model input.
