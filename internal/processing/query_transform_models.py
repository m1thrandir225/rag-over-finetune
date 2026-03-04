from dataclasses import dataclass, field


@dataclass
class TransformStageResult:
    """Outcome of a single transform stage."""

    transform: str
    generated_queries: list[str]
    duration_ms: float


@dataclass
class TransformResult:
    """
    Full result of the query-transformation pipeline.

    Attributes
    ----------
    original_query : str
        The verbatim user question.
    expanded_queries : list[str]
        All queries to use for retrieval (always includes original unless
        gated out).  For HyDE the hypothetical doc text is here so the
        chain can embed it.
    use_embedding_of : set[int]
        Indices into *expanded_queries* whose text should be embedded
        rather than searched as-is (the HyDE path).
    applied_transforms : list[str]
        Names of transforms that actually ran.
    stages : list[TransformStageResult]
        Per-stage breakdown (for observability).
    gate_skipped : bool
        True when the gate decided transforms were unnecessary.
    total_duration_ms : float
        Wall time of the full pipeline.
    """

    original_query: str
    expanded_queries: list[str]
    use_embedding_of: set[int] = field(default_factory=set)
    applied_transforms: list[str] = field(default_factory=list)
    stages: list[TransformStageResult] = field(default_factory=list)
    gate_skipped: bool = False
    total_duration_ms: float = 0.0
