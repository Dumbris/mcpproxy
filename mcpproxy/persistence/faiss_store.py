"""Faiss vector store operations."""

import sys
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]


class FaissStore:
    """Faiss vector store for tool embeddings."""

    def __init__(self, index_path: str = "tools.faiss", dimension: int = 384, initial_next_id: int | None = None):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.next_id = initial_next_id if initial_next_id is not None else 0
        # Ensure parent directory exists
        if self.index_path.parent != Path('.'):
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_index()

    def _init_index(self) -> None:
        """Initialize or load Faiss index."""
        try:
            import faiss  # type: ignore[import-untyped]
        except ImportError:
            print(
                "\n❌ ERROR: Vector storage requires faiss-cpu but it's not installed.",
                file=sys.stderr
            )
            print("   To use vector storage, install with:", file=sys.stderr)
            print(
                "   pip install mcpproxy[huggingface] or pip install mcpproxy[openai]",
                file=sys.stderr
            )
            print("   or pip install faiss-cpu", file=sys.stderr)
            print(file=sys.stderr)
            sys.exit(1)

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                # Check if dimension matches
                if self.index.d != self.dimension:
                    print(
                        f"Warning: Existing index dimension {self.index.d} != expected {self.dimension}",
                        file=sys.stderr
                    )
                    print("Recreating index with correct dimension...", file=sys.stderr)
                    self.index_path.unlink()  # Remove old index
                    self.index = faiss.IndexFlatL2(self.dimension)
                    # If index is recreated, ensure next_id is at least the initial value (or 0)
                    self.next_id = max(self.next_id, 0)
                else:
                    # If index is loaded successfully, use its ntotal as the basis for next_id,
                    # but ensure it's at least the provided initial_next_id
                    self.next_id = max(self.next_id, self.index.ntotal)
            except Exception as e:
                print(f"Error loading existing index: {e}. Creating new index...", file=sys.stderr)
                self.index_path.unlink(missing_ok=True)
                self.index = faiss.IndexFlatL2(self.dimension)
                # If new index is created due to error, ensure next_id is at least the initial value (or 0)
                self.next_id = max(self.next_id, 0)
        else:
            # Using IndexFlatL2 for exact search (can be changed to IndexIVFFlat for approximate)
            self.index = faiss.IndexFlatL2(self.dimension)
            # If new index, ensure next_id is at least the initial value (or 0)
            self.next_id = max(self.next_id, 0)

    async def add_vector(self, vector: np.ndarray) -> int:
        """Add a vector to the index and return its ID."""
        if self.index is None: # Add guard for None
            raise RuntimeError("Faiss index not initialized.")

        # Ensure vector is the right shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Add to index
        self.index.add(vector)
        vector_id = self.next_id
        self.next_id += 1

        # Save index
        await self._save_index()
        return vector_id

    async def update_vector(self, vector_id: int, vector: np.ndarray) -> None:
        """Update a vector in the index.

        Note: Faiss doesn't support direct updates, so we reconstruct the index.
        For large indices, consider using remove/add pattern.
        """
        # For simplicity, we'll rebuild the entire index
        # In a production system, you might want to implement a more efficient approach

        # This is a simplified implementation - in practice you'd want to
        # store vectors separately and rebuild the index periodically
        pass

    async def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> list[tuple[int, float]]:
        """Search for similar vectors."""
        if self.index is None or self.index.ntotal == 0: # Add guard for None and empty index
            return []

        # Ensure query vector is the right shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        # Convert to list of (index, distance) tuples
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index != -1:  # Valid result
                # Convert distance to similarity score (1 / (1 + distance))
                similarity = 1.0 / (1.0 + float(distance))
                results.append((int(index), similarity))

        return results

    async def remove_vectors(self, vector_ids: list[int]) -> None:
        """Remove vectors from the index.

        Note: Faiss doesn't support direct removal, so this is a placeholder.
        In practice, you'd need to rebuild the index without these vectors.
        """
        # This would require rebuilding the index
        # For now, we'll just note that this is not implemented
        pass

    async def _save_index(self) -> None:
        """Save the index to disk."""
        try:
            import faiss  # type: ignore[import-untyped]
            
            # Ensure parent directory exists before saving
            if self.index_path.parent != Path('.'):
                self.index_path.parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            print(f"Warning: Could not save index: {e}", file=sys.stderr)

    async def get_vector_count(self) -> int:
        """Get the number of vectors in the index."""
        return self.index.ntotal if self.index else 0

    async def reset(self) -> None:
        """Reset the index (remove all vectors)."""
        try:
            import faiss  # type: ignore[import-untyped]

            self.index = faiss.IndexFlatL2(self.dimension)
            self.next_id = 0

            # Remove index file if it exists
            if self.index_path.exists():
                self.index_path.unlink()
        except Exception as e:
            print(f"Warning: Could not reset index: {e}", file=sys.stderr)

    async def close(self) -> None:
        """Save and close index."""
        await self._save_index()
