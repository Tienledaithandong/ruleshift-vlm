# VLM Decision Reference

**Reference version:** vlm-decision-reference-v1.
**Implementation:** src/ruleshift/evaluator/vlm_decision.py.
**Status:** CPU reference and evaluator-side labels only. No model inference.

## 1. Scope

This document specifies the exact macro reference used to label the
probe-versus-commit development benchmark. It is deliberately small enough to
audit by hand and exact enough to be reproduced with the finite-horizon dynamic
program in the implementation.

The reference values a **local macro decision**. It does not model primitive
navigation mistakes, partial observability during movement, or the full
RuleShift episode. Navigation to the chosen probe or committed goal is delegated
to the public planner. The benchmark therefore measures decision selection, not
low-level control.

## 2. Notation

For a context c:

- R = six candidate rule ids in the public order.
- p = public posterior over R, a probability vector.
- b = remaining tool units.
- s, g = flags for whether the sign and blue probes were already used.
- k = probes left in the finite horizon.
- epsilon = blue inspection error.
- c_s, c_b = sign and blue tool costs.
- M(p) = the MAP rule under p, with deterministic candidate-order tie-breaking.
- G(M(p)) = the public goal colour of the MAP rule.
- T = target marker for that colour.
- n_r = public commit step count for candidate rule r.
- step_cost = 0.05 per navigation step.
- success_reward = 10.0 for a goal role; hazard_penalty = -2.0 for a hazard
  role; safe role payoff = 0.0.

The public posterior is already conditioned on the observed evidence. The
reference never conditions on hidden truth.

## 3. Commit value

Let role(r, G) be the role of the committed goal colour under candidate rule r.
Then:

    Q_commit(p) = sum_r p[r] * payoff(role(r, G(M(p))))
                  - step_cost * n_{M(p)}

where payoff is success_reward, hazard_penalty, or 0.0. The commit target is
always the MAP rule's goal colour. If two rules tie for the MAP, the earlier
candidate in R wins. The result is deterministic.

A commit is always feasible, including when the tool budget is zero.

## 4. Sign observation

For a context with a public signpost and adjacent face, each candidate rule
deterministically emits one clause for that face. The distinct clauses and their
posterior masses are:

    P(clause j) = sum_{r: clause_c(r) = j} p[r]

After observing clause j, condition exactly:

    p_j[r] = p[r] * 1[clause_c(r) = j] / P(clause j)

Because the sign is noise-free on the current development layouts, the
observation can collapse the posterior to a subset. If all candidates emit the
same clause, the probe is uninformative.

## 5. Blue observation

Let role_c(r) be the true role of the blue cell under rule r. The observed role
is one of goal, hazard, or safe. The likelihood is:

    P(observed = o | role_c(r) = t) = 1 - epsilon   if o = t
                                       epsilon / 2  otherwise

After observing o:

    p_o[r] = p[r] * P(o | role_c(r)) / sum_q p[q] * P(o | role_c(q))

At epsilon = 0, every likelihood is zero or one; observations that are
impossible under the prior are not enumerated. At epsilon > 0, all three roles
are possible.

## 6. Finite-horizon recurrence

Define V(c, p, b, s, g, k) as the maximum expected control value at a state.
If k <= 0:

    V = Q_commit(p)

Otherwise:

    V = max(
          Q_commit(p),
          Q_sign(p, b, s, g, k) if the sign is feasible,
          Q_blue(p, b, s, g, k) if blue is feasible
        )

with:

    Q_sign = -c_s + sum_j P(clause j) * V(p_j, b - c_s, true, g, k - 1)

    Q_blue = -c_b + sum_o P(o) * V(p_o, b - c_b, s, true, k - 1)

A probe is feasible only when it has not been used and its cost is at most b.
Each probe can be used at most once. The root reference computes Q_commit,
Q_sign, and Q_blue separately with the same recurrence, then compares them.

The published development contexts use max_probes = 2 and a budget of 0 to 4
units. A probe followed by a second probe is therefore possible in some states;
the recurrence values that option correctly instead of treating each probe in
isolation.

## 7. Tie handling and decision classes

Values within 1e-9 are ties. The canonical order is:

    COMMIT, PROBE_SIGN, PROBE_BLUE

The deterministic best action is the first action in that order whose value is
within tolerance of the maximum. The full optimal_actions tuple is retained, so
tie-aware scoring does not punish a different but equally optimal action.

Decision classes:

- tie: at least two feasible actions are within tolerance;
- commit_preferred: COMMIT is the unique best action and the margin is greater
  than the near-tie margin;
- probe_preferred: a unique probe is best and the margin is greater than the
  near-tie margin;
- near_tie: a unique best action exists, but the margin is at most 0.25.

A forced commit (only COMMIT is feasible) is commit_preferred, not a near tie.
This matters for budget-exhausted contexts.

The decision margin is best_value minus the second-best feasible value. For a
forced commit the margin is zero by construction, but the class is not near_tie.

## 8. Reference label fields

Each label contains:

- context_id and reference_version;
- q_commit, q_sign, q_blue (the latter two are null when infeasible);
- best_value, optimal_actions, best_action, decision_class, decision_margin;
- commit_rule, commit_target, commit_steps;
- latent_rule;
- declaration action, declared rule, confidence, margin, correctness;
- posterior_entropy.

The latent rule is sampled deterministically from the public posterior using a
SHA-256 seed derived from context_id. It exists only to make declaration scoring
possible on a synthetic fixture. It is not a real simulator truth, and a
declaration is scored against this draw, not against a hidden map that the
benchmark actually executed.

## 9. Declaration gate

Declaration is separate from control. The reference gate is:

    declare the MAP rule iff top_probability >= 0.90
                            and top_probability - runner_up_probability >= 0.10
    otherwise abstain.

Confidence is the posterior mass of the declared rule. The gate is a reference
policy for coverage and calibration, not an oracle. Its own accuracy on the
development fixture is 42/44 = 0.9545 because the latent rule is sampled.

## 10. Metrics

Decision scoring:

- **optimal_action_accuracy**: fraction of all contexts whose prediction is in
  the tie-aware optimal set. An invalid prediction is never counted as optimal;
  the denominator remains all contexts.
- **mean regret**: mean of max(0, best_value - value(predicted action)).
- **over_probe_rate**: fraction where a probe was predicted but the reference
  best action is COMMIT.
- **under_probe_rate**: fraction where COMMIT was predicted but the reference
  best action is a probe.
- **tool_selection_accuracy**: among probe predictions, fraction whose probe is
  in the optimal set.
- **invalid_rate**: fraction with an action outside the menu or infeasible
  under the remaining budget.
- **fallback_rate**: fraction marked as a parser/backend fallback by the arm.

Declaration scoring:

- **coverage**: declared, non-invalid rules divided by all contexts.
- **accuracy_given_declared**: correct declarations divided by declared.
- **abstention_rate**: ABSTAIN actions divided by all contexts.
- **invalid_rate**: declaration actions outside the menu or rules outside the
  candidate menu.
- **ECE (10 bins)**: confidence calibration over declared rows only.

The declaration metrics are never combined into the control regret. A model
cannot buy a better control score by declaring a rule.

## 11. Known reference checks

These checks are deterministic and are covered by tests:

| Context | q_commit | q_sign | q_blue | Best |
| --- | ---: | ---: | ---: | --- |
| base/hazard 50-50, default, no evidence | 3.90 | 8.925 | 7.60 | PROBE_SIGN |
| base/remap 50-50, default, no evidence | 9.90 | 8.90 | 7.90 | COMMIT |
| hazard/sign 50-50, default, no evidence | 4.95 | 6.52625 | 7.52625 | PROBE_BLUE |
| base/hazard 50-50, exhausted, no evidence | 3.90 | null | null | COMMIT |

The first row shows a sign probe that separates red-goal from red-hazard. The
second shows an uninformative sign and a commit. The third shows a sign that is
uninformative about blue/safe while blue is decisive. The fourth shows a forced
commit with no probe option.

## 12. Numerical and modeling limitations

- The reference is exact only for the stated macro model. It is not an exact
  solver for the full grid environment.
- The public posterior is an input. The benchmark does not test belief
  construction from raw observations; it tests decisions given a public belief.
- The sign is noise-free. The blue tool is the only noisy observation.
- Commit step counts come from the public planner and are deterministic; path
  execution failures are outside this reference.
- Only one commit target (the MAP target) is considered. A more general planner
  could consider alternative commit targets; that would be a different
  benchmark.
- The latent declaration rule is a posterior draw, not a true rule. Declaration
  accuracy is a fixture property and must not be presented as real-world rule
  recovery.
- Ties and near ties are intentionally visible. A model that is accurate only on
  the easy commit-preferred majority is not a good decision-maker.

## 13. Reproducibility

Contexts are sorted by context_id. Selection uses a SHA-256 hash of the context
id, not a random seed. Public JSONL, labels JSONL, audit JSON, and manifest
hashes are produced by the builder. Tests assert that reference labels and
context ids are stable across repeated builds in a process.
