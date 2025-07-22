"""Indexer and search facade."""

from typing import Any

from ..models.schemas import EmbedderType, SearchResult, ToolMetadata
from ..persistence.facade import PersistenceFacade
from ..utils.dependencies import check_embedder_dependencies
from ..utils.hashing import compute_tool_hash
from .base import BaseEmbedder
from .bm25 import BM25Embedder


class IndexerFacade:
    """Facade for indexing and searching tools."""

    def __init__(
        self,
        persistence: PersistenceFacade,
        embedder_type: EmbedderType = EmbedderType.BM25,
        hf_model: str | None = None,
        index_dir: str | None = None,
    ):
        self.persistence = persistence
        self.embedder_type = embedder_type
        self.index_dir = index_dir
        self.embedder = self._create_embedder(embedder_type, hf_model, index_dir)
        self._needs_reindex = False

    def _create_embedder(
        self, embedder_type: EmbedderType, hf_model: str | None, index_dir: str | None
    ) -> BaseEmbedder:
        """Create embedder based on type."""
        # Check dependencies before creating embedder
        check_embedder_dependencies(embedder_type.value)

        if embedder_type == EmbedderType.BM25:
            embedder = BM25Embedder(index_dir)
            # Try to load existing index
            if hasattr(embedder, "load_index"):
                embedder.load_index()
            return embedder
        elif embedder_type == EmbedderType.HF:
            from .huggingface import HuggingFaceEmbedder

            model_name = hf_model or "sentence-transformers/all-MiniLM-L6-v2"
            return HuggingFaceEmbedder(model_name)
        elif embedder_type == EmbedderType.OPENAI:
            from .openai import OpenAIEmbedder

            return OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")

    async def index_tool(
        self,
        name: str,
        description: str,
        server_name: str,
        params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        annotations: Any | None = None,
    ) -> None:
        """Index a tool with its metadata using Tool class fields."""
        # Include tags and annotations in hash computation
        extended_params = {
            "parameters": params or {},
            "tags": tags or [],
            "annotations": annotations,
        }
        tool_hash = compute_tool_hash(name, description, extended_params)

        # Check if tool already exists
        existing_tool = await self.persistence.get_tool_by_hash(tool_hash)
        if existing_tool:
            return  # Tool unchanged, no need to re-index

        # Create enhanced text for embedding (include tags)
        enhanced_text = self.embedder.combine_tool_text(name, description, params)
        if tags:
            enhanced_text += f" | Tags: {', '.join(tags)}"

        # For BM25, we handle indexing differently
        if isinstance(self.embedder, BM25Embedder):
            # Add to corpus for later batch indexing
            vector = await self.embedder.embed_text(enhanced_text)
            self._needs_reindex = True
        else:
            # For vector embedders, create embedding immediately
            vector = await self.embedder.embed_text(enhanced_text)

        # Store in persistence layer with extended metadata
        tool = ToolMetadata(
            name=name,
            description=description,
            hash=tool_hash,
            server_name=server_name,
            params_json=str(extended_params) if extended_params else None,
        )

        await self.persistence.store_tool_with_vector(tool, vector)

    async def index_tool_from_object(self, tool_obj: Any, server_name: str) -> None:
        """Index a tool from a Tool object, using hash-based change detection."""
        from fastmcp.tools.tool import Tool

        if not isinstance(tool_obj, Tool):
            raise ValueError(f"Expected Tool object, got {type(tool_obj)}")

        # Compute hash from all tool fields for change detection
        tool_data = {
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters,
            "tags": list(tool_obj.tags) if tool_obj.tags else [],
            "annotations": str(tool_obj.annotations)
            if tool_obj.annotations
            else None,
        }
        tool_hash = compute_tool_hash(
            tool_obj.name, tool_obj.description or "", tool_data
        )

        # Check if tool already exists with same hash
        existing_tool = await self.persistence.get_tool_by_hash(tool_hash)
        if existing_tool:
            return  # Tool unchanged, no need to re-index

        # Create enhanced text for embedding (include tags)
        enhanced_text = self.embedder.combine_tool_text(
            tool_obj.name, tool_obj.description or "", tool_obj.parameters
        )
        if tool_obj.tags:
            enhanced_text += f" | Tags: {', '.join(tool_obj.tags)}"

        # For BM25, we handle indexing differently
        if isinstance(self.embedder, BM25Embedder):
            # Add to corpus for later batch indexing
            vector = await self.embedder.embed_text(enhanced_text)
            self._needs_reindex = True
        else:
            # For vector embedders, create embedding immediately
            vector = await self.embedder.embed_text(enhanced_text)

        # Store minimal metadata in persistence layer (only for hash-based change detection)
        import json

        tool = ToolMetadata(
            name=tool_obj.name,
            description=tool_obj.description or "",
            hash=tool_hash,
            server_name=server_name,
            params_json=json.dumps(tool_obj.parameters) if tool_obj.parameters else None,
        )

        await self.persistence.store_tool_with_vector(tool, vector)

    async def reindex_all_tools(self) -> None:
        """Rebuild the entire index with all stored tools (BM25 specific)."""
        if not isinstance(self.embedder, BM25Embedder):
            return  # Only applicable for BM25

        # Get all tools from persistence
        all_tools = await self.persistence.get_all_tools()
        if not all_tools:
            return

        # Create text representations for all tools
        texts = []
        for tool in all_tools:
            import json

            try:
                extended_params = (
                    json.loads(tool.params_json) if tool.params_json else {}
                )
            except (json.JSONDecodeError, TypeError):
                extended_params = {}

            params = (
                extended_params.get("parameters", {})
                if isinstance(extended_params, dict)
                else None
            )
            tags = (
                extended_params.get("tags", [])
                if isinstance(extended_params, dict)
                else []
            )

            text = self.embedder.combine_tool_text(tool.name, tool.description, params)
            if tags:
                text += f" | Tags: {', '.join(tags)}"
            texts.append(text)

        # Rebuild BM25 index with all tools
        await self.embedder.fit_corpus(texts)
        self._needs_reindex = False

    async def reset_embedder_data(self) -> None:
        """Reset embedder-specific data."""
        if isinstance(self.embedder, BM25Embedder):
            await self.embedder.reset_stored_data()
        # For other embedders, no additional reset needed beyond persistence reset

    async def ensure_index_ready(self) -> None:
        """Ensure the index is ready for search (reindex if needed)."""
        if isinstance(self.embedder, BM25Embedder):
            # If we need reindexing or the embedder has no corpus
            if self._needs_reindex or not self.embedder.is_indexed():
                await self.reindex_all_tools()

    async def search_tools(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for tools using the configured embedder."""
        # Ensure index is ready for BM25
        await self.ensure_index_ready()

        if isinstance(self.embedder, BM25Embedder):
            # For BM25, use direct search method
            return await self._search_with_bm25(query, k)
        else:
            # For vector embedders, use vector similarity
            return await self._search_with_vectors(query, k)

    async def _search_with_vectors(self, query: str, k: int) -> list[SearchResult]:
        """Search using vector similarity."""
        query_vector = await self.embedder.embed_text(query)
        return await self.persistence.search_similar_tools(query_vector, k)

    async def _search_with_bm25(self, query: str, k: int) -> list[SearchResult]:
        """Search using BM25 similarity."""
        # Get all tools and their texts
        all_tools = await self.persistence.get_all_tools()
        if not all_tools:
            return []

        # Use the BM25 embedder's search method with the indexed corpus
        results = await self.embedder.search_similar(query, None, k)

        if not results:
            return []

        # Extract scores for normalization
        scores = [score for _, score in results]

        # Use modified sigmoid normalization to map scores to [0,1] range
        # Handle zero scores (nonsense queries) specially
        import math

        normalized_scores = []
        for score in scores:
            if score <= 0.0:
                # Map zero/negative scores to very low values (< 0.5)
                normalized_score = 0.1
            else:
                # Apply sigmoid function: 1 / (1 + exp(-score))
                # This maps positive scores to (0.5, 1) range
                normalized_score = 1.0 / (1.0 + math.exp(-score))
            normalized_scores.append(normalized_score)

        # Convert to SearchResult objects with normalized scores
        search_results = []
        for i, (idx, _) in enumerate(results):
            if idx < len(all_tools):
                search_results.append(
                    SearchResult(tool=all_tools[idx], score=normalized_scores[i])
                )

        return search_results

    async def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the current index."""
        stats = {
            "embedder_type": self.embedder_type.value,
            "needs_reindex": self._needs_reindex,
        }

        if isinstance(self.embedder, BM25Embedder):
            stats.update(
                {
                    "corpus_size": len(self.embedder.corpus),
                    "indexed": self.embedder.indexed,
                    "index_dir": self.embedder.index_dir,
                }
            )
        else:
            # For vector embedders
            stats.update(
                {
                    "dimension": self.embedder.get_dimension(),
                }
            )

        return stats
