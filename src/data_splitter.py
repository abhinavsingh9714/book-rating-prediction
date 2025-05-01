import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle


def split_train_test_sparse(sparse_matrix, test_ratio=0.25, seed=42):
    """
    Splits the sparse user-item matrix into a training matrix (same shape),
    with test set stored as a list of (user, item, rating) tuples.

    Args:
        sparse_matrix (csr_matrix): Full user-item sparse matrix
        test_ratio (float): Fraction of ratings to hold out for testing
        seed (int): Random seed for reproducibility

    Returns:
        train_matrix (csr_matrix): Sparse matrix with only training ratings
        test_set (list of tuples): [(user_idx, item_idx, rating), ...]
    """
    np.random.seed(seed)

    # Get non-zero indices
    user_ids, item_ids = sparse_matrix.nonzero()
    ratings = sparse_matrix[user_ids, item_ids].A1  # Extract actual values as flat array

    # Create triplets
    data = list(zip(user_ids, item_ids, ratings))

    # Shuffle and split
    data = shuffle(data, random_state=seed)
    cutoff = int(len(data) * (1 - test_ratio))
    train_data = data[:cutoff]
    test_data = data[cutoff:]

    print(f"📊 Total ratings: {len(data)}")
    print(f"🧪 Test set size: {len(test_data)}")
    print(f"🧩 Training set size: {len(train_data)}")

    # Build training sparse matrix
    train_user_ids, train_item_ids, train_ratings = zip(*train_data)
    train_matrix = csr_matrix(
        (train_ratings, (train_user_ids, train_item_ids)),
        shape=sparse_matrix.shape
    )

    return train_matrix, test_data

