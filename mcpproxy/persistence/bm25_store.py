"""Simple BM25 store that doesn't require faiss."""

import numpy as np


class BM25Store:
    """Simple store for BM25 that doesn't use faiss vectors."""

    def __init__(self, index_path: str = "bm25_store", dimension: int = 1, initial_next_id: int | None = None):
        """Initialize BM25 store.
        
        Args:
            index_path: Not used for BM25, kept for API compatibility
            dimension: Not used for BM25, kept for API compatibility  
        """
        self.index_path = index_path
        self.dimension = dimension
        self.next_id = initial_next_id if initial_next_id is not None else 0

    async def add_vector(self, vector: np.ndarray) -> int:
        """Add a placeholder vector and return its ID.
        
        For BM25, we don't actually store vectors since BM25 handles its own indexing.
        This just returns a sequential ID for database compatibility.
        """
        vector_id = self.next_id
        self.next_id += 1
        return vector_id

    async def update_vector(self, vector_id: int, vector: np.ndarray) -> None:
        """Update a vector (no-op for BM25)."""
        pass

    async def search(self, query_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors (not used for BM25).
        
        Returns empty arrays since BM25 uses its own search mechanism.
        """
        return np.array([]), np.array([])

    async def remove_vectors(self, vector_ids: list[int]) -> None:
        """Remove vectors (no-op for BM25)."""
        pass

    async def get_vector_count(self) -> int:
        """Get the number of vectors (always 0 for BM25)."""
        return 0

    async def reset(self) -> None:
        """Reset the store."""
        self.next_id = 0

    async def close(self) -> None:
        """Close store (no-op for BM25)."""
        pass 
