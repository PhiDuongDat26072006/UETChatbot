"""
src/graph - Phân hệ Knowledge Graph & LightRAG khép kín cho UET Chatbot.
Heavy dependencies load on demand via __getattr__.
"""

__all__ = [
    "AcademicThesaurus",
    "get_thesaurus",
    "FORBIDDEN_ENTITIES",
    "EXTENDED_FORBIDDEN_ENTITIES",
    "is_forbidden_entity",
    "AdaptiveSemanticExtractor",
    "SemanticUnit",
    "LightRAGIndexer",
    "GraphIndex",
    "LightRAGChunkBridge",
    "RegexSupplementGraph",
    "HybridGraphExtractor",
    "QueryEntityExtractor",
    "LightRAGRetriever",
]


def __getattr__(name: str):
    if name in (
        "AcademicThesaurus",
        "get_thesaurus",
        "FORBIDDEN_ENTITIES",
        "EXTENDED_FORBIDDEN_ENTITIES",
        "is_forbidden_entity",
    ):
        from src.graph import thesaurus_utils

        return getattr(thesaurus_utils, name)
    if name in ("AdaptiveSemanticExtractor", "SemanticUnit"):
        from src.graph import adaptive_extractor

        return getattr(adaptive_extractor, name)
    if name == "LightRAGIndexer":
        from src.graph.lightrag_indexer import LightRAGIndexer

        return LightRAGIndexer
    if name in (
        "GraphIndex",
        "LightRAGChunkBridge",
        "RegexSupplementGraph",
        "HybridGraphExtractor",
    ):
        from src.graph import graph

        return getattr(graph, name)
    if name == "QueryEntityExtractor":
        from src.graph.query_entity_extractor import QueryEntityExtractor

        return QueryEntityExtractor
    if name == "LightRAGRetriever":
        from src.graph.lightrag_retriever import LightRAGRetriever

        return LightRAGRetriever
    raise AttributeError(name)
