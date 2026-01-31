from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """
    Result of a query from the system
    """

    answer: str
    sources: list[str] = field(default_factory=list)

    query: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
        }
