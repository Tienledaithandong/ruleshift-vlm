# -*- coding: utf-8 -*-
path = 'docs/problem_spec.md'
with open(path, 'r', encoding='utf-8', newline='') as f:
    data = f.read()

start = data.index('## 12. Proposed method constraint')
end = data.index('## 13. Acceptance tests')

new_section = (
    '## 12. Proposed method constraint (no hidden model borrowing)\n'
    '\n'
    'The planned VLM method consumes only public RGB, public history, candidate\n'
    'descriptions and public cost text. The selector may do arithmetic on\n'
    '**VLM-provided** estimates and public constants, but must not access simulator\n'
    'likelihoods, `env.copy().step`, Bayes `Q*`, oracle belief or true rules.\n'
    '\n'
    '### 12.1 M1 one-step selector — baseline/diagnostic, not proven effective\n'
    '\n'
    'The reference one-step selector (`ValueAwareSelector`, alias `M1`) uses\n'
    '\n'
    '```\n'
    "Qhat(a|h) = E_{m~b_hat, o'~Phat}[ rhat(h,a,m,o') + Vhat_1(h + (a,o')) ]\n"
    '```\n'
    '\n'
    'with `Vhat_1` a **one-step** continuation `max_a rhat(a | posterior)`. This is a\n'
    'deliberately shallow baseline, added so that probes (zero immediate reward) are\n'
    'not systematically undervalued by pure immediate-reward arithmetic.\n'
    '\n'
    '**Known limitation, documented rather than hidden:** `Vhat_1` is added to\n'
    '*every* outcome branch, including successful commits, so future commits are\n'
    'double-counted relative to a probe whose information pays off across several\n'
    'future decisions. When both probe and commit durations fit inside the horizon,\n'
    'the continuation largely cancels and the comparison collapses to probe cost\n'
    'versus commit expected immediate reward. Consequently M1 one-step can select\n'
    'an immediate commit even when the exact finite-horizon optimum is a probe\n'
    '(see `tests/test_m1_microtasks.py::test_microtask_A_probe_valuable_but_m1_picks_commit`).\n'
    '\n'
    'Because of this, M1 one-step is presented as a **baseline/diagnostic** for\n'
    'Week 1-2 development, not as a claim of an effective probe-vs-task method.\n'
    '`Delta_probe` and probe-vs-commit conclusions must not be drawn from M1 one-step\n'
    'results alone.\n'
    '\n'
    '### 12.2 Multi-step Vhat — experimental, diagnostics only\n'
    '\n'
    '`MultiStepValueAwareSelector` (ablation `M1_multi_step`) replaces `Vhat_1` with\n'
    'a finite-horizon recursion `V(b, steps_left)` over public predictions only:\n'
    'probe feedback updates the public belief and consumes `probe_duration` steps;\n'
    'a commit consumes `commit_duration` (+ lockout on reject) but the agent may\n'
    'keep acting while steps remain. No simulator likelihood, no oracle, no hidden\n'
    'transitions are used.\n'
    '\n'
    'This ablation is included in the Week 6 comparison matrix (`ABLATION_NAMES`)\n'
    'so that its behavior can be measured against M1 one-step on dev scenarios with\n'
    'hand-computed exact optima. It remains **experimental**: outcomes are assumed\n'
    'fixed from the initial VLM payload (no re-query per recursion step), and the\n'
    'belief updater is a soft heuristic that does not model feedback noise or\n'
    'staleness after a shift (see section 15 Q4).\n'
    '\n'
    '---\n'
    '\n'
)

data = data[:start] + new_section + data[end:]
with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(data)
print('Section 12 replaced successfully. New file length:', len(data))