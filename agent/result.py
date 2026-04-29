"""AgentResult — 识别结果数据结构（独立模块，无外部依赖）"""

from __future__ import annotations


class AgentResult:
    """识别运行结果，包含结构化信息供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer
