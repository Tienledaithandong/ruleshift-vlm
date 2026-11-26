"""RuleShift / Agentic VLM research repository.

Status: contracts and static interfaces (``public_api``), the Gymnasium
environments (``envs``), the privileged evaluator references (``evaluator``),
and the public-only agent pipeline (``agents``) are implemented. There is no
training and no fine-tuning; the default model backend is a deterministic stub
marked ``STUB_SYNTHETIC``, and no empirical result exists yet.

The research question is frozen in the master plan: when hidden environment
dynamics change without warning, do frozen vision-language agents choose probes
whose control value justifies their cost, and can structured counterfactual
outcome reasoning (without the generative model) reduce probe-decision regret
and improve closed-loop return?
"""

__version__ = "0.1.0"
