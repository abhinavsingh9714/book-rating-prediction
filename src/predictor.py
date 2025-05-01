# Rating prediction logic
# src/predictor.py

import numpy as np
from scipy.sparse import csr_matrix


def predict_rating(user_idx, item_idx, train_matrix, similarity_matrix, k=10):
    """
    Predicts the rating a user would give to an item using item-item collaborative filtering.

    Args:
        user_idx (int): Index of the user in the matrix.
        item_idx (int): Index of the item (book) to predict.
        train_matrix (csr_matrix): Training matrix (user-item ratings).
        similarity_matrix (csr_matrix): Precomputed item-item similarity matrix.
        k (int): Number of similar items to consider.

    Returns:
        float: Predicted rating
    """
    # Get items rated by the user
    user_ratings = train_matrix[user_idx].toarray().ravel()
    rated_items = np.where(user_ratings > 0)[0]
    # print(f"User {user_idx} rated items: {rated_items}")
    if len(rated_items) == 0:
        print(f"⚠️ User {user_idx} has not rated any items.")
        global_mean = train_matrix.data.mean()  # All ratings mean
        return global_mean
        # return 0  # No history available

    # Get similarities between target item and items rated by user
    similarities = similarity_matrix[item_idx, rated_items].toarray().ravel()
    ratings = user_ratings[rated_items]

    # Sort top k similar items
    if len(similarities) > k:
        top_k_idx = np.argsort(similarities)[::-1][:k]
        similarities = similarities[top_k_idx]
        ratings = ratings[top_k_idx]

    # Compute weighted average
    numerator = np.dot(similarities, ratings)
    denominator = np.sum(np.abs(similarities))

    if denominator == 0:
        print(f"⚠️ No similar items found for user {user_idx} and item {item_idx}.")
        if len(rated_items) > 0:
            return np.mean(ratings)  # Use user’s average rating
        else:
            global_mean = train_matrix.data.mean()  # All ratings mean
            return global_mean
        # return 0  # Avoid divide-by-zero

    return numerator / denominator

# def predict_user_user_rating(user_idx, item_idx, train_matrix, similarity_matrix, k=10):
#     """
#     Predict rating using user-user collaborative filtering.
    
#     Args:
#         user_idx (int): Target user
#         item_idx (int): Target item
#         train_matrix (csr_matrix): User-item ratings
#         similarity_matrix (csr_matrix): User-user similarity matrix
#         k (int): Top-k neighbors

#     Returns:
#         float: predicted rating
#     """
#     # Get users who rated this item
#     item_column = train_matrix[:, item_idx].toarray().ravel()
#     users_who_rated = np.where(item_column > 0)[0]
#     print(f"Users who rated item {item_idx}: {users_who_rated}")
#     if len(users_who_rated) == 0:
#         return 0  # no rating info available

#     similarities = similarity_matrix[user_idx, users_who_rated].toarray().ravel()
#     ratings = item_column[users_who_rated]

#     if len(similarities) > k:
#         top_k_idx = np.argsort(similarities)[::-1][:k]
#         similarities = similarities[top_k_idx]
#         ratings = ratings[top_k_idx]

#     numerator = np.dot(similarities, ratings)
#     denominator = np.sum(np.abs(similarities))

#     if denominator == 0:
#         return 0  # fallback needed

#     return numerator / denominator