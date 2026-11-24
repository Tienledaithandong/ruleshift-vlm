"""History-only soft belief updating for structured predictions.

Uses only the public action/outcome history and public candidate rule ids. It
never reads the simulator, the evaluator, a transition table or a true rule.

Known limitations are additive hardening, all disabled by default so existing
behaviour is preserved:
  - no feedback-noise model -> noise_floor
  - unobserved rules never decay -> relevance_decay
"""

from __future__ import annotations

from dataclasses import dataclass

Belief = dict[str, float]


@dataclass(frozen=True)
class HistoryConsistencyUpdater:
    """Softly move probability toward the rule confirmed by each public outcome."""

    rules: tuple[str, ...] = ("red", "blue", "green")
    learning_rate: float = 0.3
    noise_floor: float = 0.0
    relevance_decay: float = 0.0

    def update(self, belief: Belief, action: str, outcome: str) -> Belief:
        """Return a new belief conditioned on one public action outcome."""

        updated = dict(belief)
        if not action.startswith(("PROBE_", "COMMIT_")):
            return updated
        if outcome not in ("accept", "reject"):
            return updated

        color = action.split("_", 1)[1].lower()
        if color not in updated:
            return updated

        eta = min(1.0, max(0.0, self.learning_rate))
        if outcome == "accept":
            updated[color] *= 1.0 + eta
            for rule in self.rules:
                if rule != color:
                    updated[rule] *= 1.0 - eta
        else:
            updated[color] *= 1.0 - eta

        # Relevance decay: rules not confirmed by this outcome lose a fixed
        # fraction of mass per update, modelling staleness.
        if self.relevance_decay > 0.0:
            decay = 1.0 - self.relevance_decay
            for rule in self.rules:
                if rule != color:
                    updated[rule] *= decay

        # Noise floor: move a fraction of mass toward the uniform prior.
        if self.noise_floor > 0.0:
            mass = 1.0 - self.noise_floor
            uniform = self.noise_floor / len(self.rules)
            updated = {rule: v * mass + uniform for rule, v in updated.items()}

        cleaned = {rule: max(0.0, updated.get(rule, 0.0)) for rule in self.rules}
        total = sum(cleaned.values())
        if total <= 0.0:
            return dict(belief)
        return {rule: value / total for rule, value in cleaned.items()}
