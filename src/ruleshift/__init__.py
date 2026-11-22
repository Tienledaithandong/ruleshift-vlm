"""RuleShift / Agentic VLM research repository.

Week 1 status: this package contains only contracts, static interfaces and
permission-checking utilities. There is intentionally no environment, no Bayes
oracle, no VLM runner and no learned model in this package yet.

The research question is frozen in the master plan: when hidden environment
dynamics change without warning, do frozen vision-language agents choose probes
whose control value justifies their cost, and can structured counterfactual
outcome reasoning (without the generative model) reduce probe-decision regret
and improve closed-loop return?
"""

__version__ = "0.1.0"
