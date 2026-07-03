"""Faithful Asch conformity clone for LLMs (knowledge + line tasks)."""
from .experiment import ConformityExperiment  # noqa: F401  (registers)
from .challenge import ConformityChallengeExperiment  # noqa: F401  (registers)
from .conformity_tool import ConformityToolExperiment  # noqa: F401  (registers)
from .dialogue import ConformityDialogueExperiment  # noqa: F401  (registers)
from .causal import ConformityCausalExperiment  # noqa: F401  (registers)
