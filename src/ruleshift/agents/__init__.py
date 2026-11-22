"""Agent-side package.

Contains public-only policies, a VLM request/response interface and cache,
history-only consistency updates and the value-aware selector. There is still
no learned model and no real VLM is run by default.

HARD RULE (see docs/agent_evaluator_permissions.md): nothing in this package
may import ``ruleshift.envs`` or ``ruleshift.evaluator``, nor call
``env.copy().step(...)`` to score counterfactual actions. Agents receive only
:class:`ruleshift.public_api.interface.PublicObservation` objects and return a
public macro-action id. This boundary is enforced by
``tests/test_agent_permissions.py`` in CI.
"""
