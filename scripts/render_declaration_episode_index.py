"""Render the documentary episode index from a diagnosis payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHORT = {
    'early_success_stability_gate': 'stability',
    'early_success_terminal_only_evidence': 'terminal-only',
    'early_success_insufficient_evidence': 'insufficient',
    'inspect_noise_confident_error': 'inspect-noise',
    'premature_commit_ambiguous_evidence': 'premature',
}
ORDER = {
    'early_success_stability_gate': 0,
    'early_success_terminal_only_evidence': 1,
    'early_success_insufficient_evidence': 2,
    'inspect_noise_confident_error': 3,
    'premature_commit_ambiguous_evidence': 4,
}


def render(payload: dict[str, Any]) -> str:
    traces = list(payload['traces'])
    v4 = {t['scenario_id']: t for t in traces if t['method'] == 'diagnostic_belief_v4'}
    v3 = {t['scenario_id']: t for t in traces if t['method'] == 'diagnostic_belief_v3'}
    if len(v4) != 37 or len(v3) != 37:
        raise ValueError('expected 37 v4 and 37 matched v3 traces')

    lines = [
        '# Declaration failure episode index',
        '',
        'This appendix indexes every one of the 37 v4 target episodes (33 unresolved, 4 incorrect) and its matched v3 run. The complete per-decision public posterior, action, tool target, tool value, remaining budget, safe-commit eligibility, and declaration event for all 74 traces are in `docs/declaration_failure_traces_v1.jsonl`.',
        '',
        'The table is a reading aid, not a substitute for the trace. `Crit` is the number of decisions at which top >= 0.90 and margin >= 0.10; `Elig` is the number at which the frozen policy safe-commit branch returned true. The v4 unresolved cases all have `Elig = 0`.',
        '',
        '## Unresolved episodes',
        '',
        '| Episode | True | Primary cause | Steps | Crit/Elig | Final top / margin / run | Tool units | v3 status | v3 steps |',
        '|---|---|---|---:|---:|---:|---:|---|---:|',
    ]
    unresolved = sorted(
        (t for t in v4.values() if t['status'] == 'unresolved'),
        key=lambda t: (ORDER[t['diagnosis']['primary']], t['scenario_id']),
    )
    for trace in unresolved:
        matched = v3[trace['scenario_id']]
        crit = sum(
            1
            for d in trace['decisions']
            if d['top'] >= 0.90 and d['margin'] >= 0.10
        )
        eligible = sum(1 for d in trace['decisions'] if d['safe_commit_eligible'])
        episode = trace['scenario_id'].replace('-identification-', ' / ').replace('-None', '')
        terminal = trace['terminal']
        lines.append(
            f'| {episode} | {trace["true_rule"]} | '
            f'{SHORT[trace["diagnosis"]["primary"]]} | {trace["env_steps"]} | '
            f'{crit}/{eligible} | {terminal["top"]:.4f} / {terminal["margin"]:.4f} / {terminal["run"]} | '
            f'{terminal["tool_units"]} | {matched["status"]} | {matched["env_steps"]} |'
        )
    lines.extend([
        '',
        'Summary: 14 stability, 11 terminal-only, 8 insufficient. The 14 stability cases were all v3 correct declarations. The 11 terminal-only cases split 5 v3 correct / 6 v3 unresolved. The 8 insufficient cases split 4 v3 correct / 4 v3 unresolved.',
        '',
        '## Incorrect declarations',
        '',
        '| Episode | True | Primary cause | Declared | Declared step | Confidence | v4 steps / termination | Tool units | v3 status |',
        '|---|---|---|---|---:|---:|---|---:|---|',
    ])
    wrong = sorted(
        (t for t in v4.values() if t['status'] == 'wrong_commit'),
        key=lambda t: t['scenario_id'],
    )
    for trace in wrong:
        matched = v3[trace['scenario_id']]
        event = trace['events'][-1]
        episode = trace['scenario_id'].replace('-identification-', ' / ').replace('-None', '')
        lines.append(
            f'| {episode} | {trace["true_rule"]} | '
            f'{SHORT[trace["diagnosis"]["primary"]]} | {event["declared_rule"]} | '
            f'{event["step"]} | {event["confidence"]:.4f} | '
            f'{trace["env_steps"]} / {trace["termination_reason"]} | '
            f'{trace["terminal"]["tool_units"]} | {matched["status"]} |'
        )
    lines.extend([
        '',
        'The same four episodes are the final incorrect declarations in v3 and v4, with the same declared rules. In v4 two are explained by a contradictory noisy `INSPECT` (the policy likelihood includes the public 5% error rate) and two are premature commitments on ambiguous movement/sign evidence; the two premature v4 cases recovered to the true terminal MAP. v3 recovered one of those two and ended `m2-s2` at the wrong MAP after the decision limit. Full pre-declaration evidence is in the JSONL appendix.',
        '',
        '## Reading the trace file',
        '',
        'Each JSONL row contains:',
        '',
        '- `scenario_id`, `scenario_type`, `map_seed`, `run_seed`, `method`, `status`, `true_rule`, `task_success`, `env_steps`, `termination_reason`, `terminal_map`, `terminal_posterior`;',
        '- `events`: time-stamped declarations emitted at the safe-commit branch;',
        '- `decisions`: one record per `choose()` call, with the complete posterior, top two probabilities, margin, leader run, safe-commit eligibility, remaining tool budget, action, target, target color, observed and true tool values, and any event emitted at that decision;',
        '- `diagnosis`: the primary cause and secondary factors assigned by the pure classifier.',
        '',
        'Terminal MAP is recorded separately from declarations. It is never counted as a declaration.',
    ])
    return chr(10).join(lines) + chr(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/declaration_diagnosis_v1.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/DECLARATION_FAILURE_EPISODES.md'))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    args.out.write_text(render(payload), encoding='utf-8')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
