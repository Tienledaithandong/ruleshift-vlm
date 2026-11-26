# v0.5 ablation report

Status: development-only analysis of the official 300-episode run at
results/grid_v05_official_final. Candidate commit
751a3414ad8815087f64d85ad5a1523af11077f4. This report does not authorize a
confirmatory claim.

## 1. Arms

- A (historical): diagnostic_belief_v3 and repaired diagnostic_belief_v4 under
  their original protocol v1. Descriptive comparator only.
- B (v4_readout_v2): repaired v4 control, separated v2 declaration readout, no
  active evidence requests.
- C (v5): candidate; same control and readout as B plus active
  REQUEST_EVIDENCE.
- D (v5_no_request): candidate class with active requests disabled. By
  construction D is behaviourally identical to B.

## 2. Exact B/C/D equivalence

On all 300 official episodes:

| check | B | C | D | mismatches |
|---|---:|---:|---:|---:|
| task successes | 298 | 298 | 298 | 0 |
| terminal MAP correct | 297 | 297 | 297 | 0 |
| final correct declarations | 289 | 289 | 289 | 0 |
| final wrong declarations | 3 | 3 | 3 | 0 |
| unresolved | 8 | 8 | 8 | 0 |
| strict clean | 282 | 282 | 282 | 0 |
| tool units | 383 | 383 | 383 | 0 |
| decision count, max | 62 | 62 | 62 | 0 |
| action-trace hash | 9a8fbad7... | 9a8fbad7... | 9a8fbad7... | 0 |
| declaration events | 299 | 299 | 299 | 0 |

The action-trace hash is the SHA-256 of the complete 300-episode chosen-action,
tool-target, and tool-value sequence:
9a8fbad7d6c151271354f70e537c918bc3bb1ab8a5974a713538fa36a758d326.
All 300 action sequences, all public tool events, and all declaration events
are byte-identical across B, C, and D. Public task outcomes and terminal MAP
are identical as well.

C differs from B only in its epistemic request log and its abstention log:

- C: 445 request events; 0 pending; 281 executed, 84 expired, 80 unmet.
- Channels: 340 SIGN, 105 BLUE.
- Status by channel: SIGN 179 executed, 81 expired, 80 unmet; BLUE 102
  executed, 3 expired, 0 unmet.
- Episodes with requests: all 300. Distribution: 1 request on 177 episodes,
  2 on 111, 3 on 7, 4 on 2, 5 on 1, 6 on 2; maximum 6.
- Abstention events: B 522, C 365, D 522. C has 157 fewer recorded abstention
  events because a request can occupy the epistemic slot that would otherwise
  produce an abstention; this is not a one-to-one request count.

The 281 executed requests correspond to ordinary BLUE/READ_SIGN probes already
selected by the v4 control layer, so the request channel did not add an action,
tool unit, or observation. It also did not add a declaration: C and B declare
the same rule on every episode.

## 3. Request lifecycle and interface checks

C requests are public directives, not actions. A request is written pending and
becomes:

- executed when the subsequent control action satisfies it and pays the normal
  public cost;
- unmet when the requested probe is not feasible or cannot be legally taken;
- expired when a declaration or terminal finalization closes the request
  before execution.

The official run contains no pending requests at the end of an episode. The
requests that were executed account for the same tool events as B; the action
trace hash equality is the strongest check that the interface did not create a
free observation. The policy tests additionally check that a request consumes
no budget, that requests cannot access evaluator state, and that declaration
references cannot be terminal-MAP conversions.

SIGN requests use the protocol's symbolic target next_informative_face; BLUE
requests use public map coordinates (for example [2, 4] and [5, 5]). Both are
derived from the public observation and are in the allowed v2 request target
vocabulary.

## 4. Interpretation

### 4.1 Separation and readout

B is the clean readout comparison: it keeps repaired v4's control decisions and
adds the separated v2 declaration logic plus the one final-observation
opportunity. Against v4, B adds 26 correct final declarations, removes 25
unresolved, and reduces final wrong by one, with task success and terminal MAP
unchanged. This supports the limited claim that a separated declaration
readout with a final-observation opportunity improves explicit identification
without changing control performance on this development slice.

The comparison is cross-protocol: v4 was scored under v1 and B under v2. It is
descriptive, not a controlled causal estimate.

### 4.2 Active evidence requests

C versus B isolates the active request component. It is exactly zero on every
control, outcome, tool, decision-count, and declaration measure. The 445
requests are therefore epistemically inert in this implementation: they are
mostly bookkeeping over probes the v4 control layer already chooses. The
experiment does not support the hypothesis that active REQUEST_EVIDENCE improves
identification beyond the separated readout. It also does not show that
requests harm performance; they simply do not change behaviour here.

The C-only abstention reduction is a logging consequence of inserting a
request decision, not evidence of better calibration.

### 4.3 D is a failed ablation

D was expected to duplicate B, and it does so exactly. It provides no
independent estimate of a no-request candidate. The request effect is estimated
by C versus B only, and that estimate is zero. D must not be presented as a
successful independent ablation or as evidence that requests are unnecessary
in general.

### 4.4 A is not a controlled v2 arm

The historical v3 and v4 arms are useful context, but they used v1 declaration
semantics and different code. Their differences from B/C/D combine protocol,
readout, and policy changes. They are not a matched causal control.

## 5. Conclusion

The v0.5 run separates two findings:

1. The separated declaration/readout and final-observation opportunity produce
   the observed identification gain over repaired v4, while preserving task and
   MAP performance.
2. The active request channel produces no behavioural change and no
   identification gain over B. The request ablation is null, and D is a
   duplicate of B rather than an independent arm.

No request-specific benefit should be claimed. The next decision is about
whether the B/C readout result is strong enough for freeze review, not about a
request-channel effect.
