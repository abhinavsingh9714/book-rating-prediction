# Item-item similarity functions
# src/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors


def compute_item_similarity(train_matrix, top_k=100):
    """
    Compute cosine similarity between items (books) using the training matrix.
    Args:
        train_matrix (csr_matrix): Sparse user-item matrix with shape (n_users, n_items)
        top_k (int): Number of top similar items to retain for each item (optional for sparsity)

    Returns:
        similarity_matrix (csr_matrix): Sparse matrix of item-item similarities
    """
    print("🧠 Computing item-item cosine similarity...")

    # Transpose: items become rows, users become columns
    item_user_matrix = train_matrix.T

    # Compute full cosine similarity matrix (dense)
    similarity_dense = cosine_similarity(item_user_matrix, dense_output=False)

    if top_k is not None:
        # Keep only top_k similarities per item (for sparsity & speed)
        print(f"🔧 Retaining top {top_k} similarities per item...")
        n_items = similarity_dense.shape[0]
        for i in range(n_items):
            row = similarity_dense[i].toarray().ravel()
            top_indices = np.argsort(row)[::-1][1:top_k + 1]  # skip self-similarity
            mask = np.ones(n_items, dtype=bool)
            mask[top_indices] = False
            similarity_dense[i, mask] = 0.0
        similarity_dense.eliminate_zeros()

    print("✅ Similarity computation complete.")
    return similarity_dense


def compute_topk_item_similarity(train_matrix, top_k=100, metric='cosine'):
    """
    Efficiently compute top-k item-item similarity using NearestNeighbors on sparse data.
    Returns a sparse similarity matrix (item-item).
    """
    print("⚡ Using NearestNeighbors to compute top-k item similarities...")

    item_user_matrix = train_matrix.T  # shape: (n_items, n_users)
    model = NearestNeighbors(n_neighbors=top_k + 1,  # +1 to include self
                             metric=metric,
                             algorithm='brute',
                             n_jobs=-1)
    model.fit(item_user_matrix)

    distances, indices = model.kneighbors(item_user_matrix)

    # Build a sparse matrix manually

    n_items = item_user_matrix.shape[0]
    similarity_matrix = lil_matrix((n_items, n_items))

    for i in range(n_items):
        for j in range(1, top_k + 1):  # skip self (index 0)
            similarity_matrix[i, indices[i, j]] = 1 - distances[i, j]  # similarity = 1 - distance

    return similarity_matrix.tocsr()

def compute_user_similarity(train_matrix, top_k=100, metric='cosine'):
    """
    Compute top-k user-user cosine similarity matrix.
    
    Args:
        train_matrix (csr_matrix): User-item matrix
        top_k (int): Number of top similar users to retain
        cache_path (str): Optional path to cache/load similarity matrix

    Returns:
        similarity_matrix (csr_matrix)
    """

    model = NearestNeighbors(n_neighbors=top_k + 1,  # +1 to include self
                             metric=metric,
                             algorithm='brute',
                             n_jobs=-1)
    model.fit(train_matrix)

    distances, indices = model.kneighbors(train_matrix)

    # Build a sparse matrix manually

    n_users = train_matrix.shape[0]
    similarity_matrix = lil_matrix((n_users, n_users))

    for i in range(n_users):
        for j in range(1, top_k + 1):  # skip self (index 0)
            similarity_matrix[i, indices[i, j]] = 1 - distances[i, j]  # similarity = 1 - distance

    return similarity_matrix.tocsr()
