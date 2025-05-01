from src.predictor import predict_rating, predict_user_user_rating
import numpy as np


def mean_absolute_difference(test_set, train_matrix, similarity_matrix, k=10):
    """
    Evaluate predictions on the test set using Mean Absolute Difference (MAD).
    
    Args:
        test_set (list of tuples): [(user_idx, item_idx, true_rating), ...]
        train_matrix (csr_matrix): Training sparse matrix with user-item ratings
        similarity_matrix (csr_matrix): Item-item similarity matrix
        k (int): Number of nearest neighbors to use in prediction
    
    Returns:
        float: Mean Absolute Difference
    """
    errors = []

    for user_idx, item_idx, true_rating in test_set:
        pred = predict_rating(user_idx, item_idx, train_matrix, similarity_matrix, k)
        # print(f"User {user_idx} predicted rating for item {item_idx}: {pred:.2f} (True: {true_rating})")
        error = abs(pred - true_rating)
        errors.append(error)

    mad = np.mean(errors)
    print(f"🎯 Mean Absolute Difference (k={k}): {mad:.4f}")
    return mad

