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
        # print(f"⚠️ User {user_idx} has not rated any items.")
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
        # print(f"⚠️ No similar items found for user {user_idx} and item {item_idx}.")
        if len(rated_items) > 0:
            return np.mean(ratings)  # Use user’s average rating
        else:
            global_mean = train_matrix.data.mean()  # All ratings mean
            return global_mean
        # return 0  # Avoid divide-by-zero

    return numerator / denominator

def predict_user_user_rating(user_idx, item_idx, train_matrix, similarity_matrix, k=10):
    """
    Predict rating using user-user CF, only considering similar users who rated the target item.
    """
    # Get users who rated the item
    item_ratings = train_matrix[:, item_idx].toarray().ravel()
    users_who_rated = np.where(item_ratings > 0)[0]

    if len(users_who_rated) == 0:
        # print(f"⚠️ No users rated item {item_idx}")
        return 0

    # Similarity between current user and those users
    user_similarities = similarity_matrix[user_idx, users_who_rated].toarray().ravel()
    ratings = item_ratings[users_who_rated]

    # Top-k filtering
    if len(user_similarities) > k:
        top_k_idx = np.argsort(user_similarities)[::-1][:k]
        user_similarities = user_similarities[top_k_idx]
        ratings = ratings[top_k_idx]

    numerator = np.dot(user_similarities, ratings)
    denominator = np.sum(np.abs(user_similarities))

    if denominator == 0:
        # print(f"⚠️ No similar users found for user {user_idx} and item {item_idx}.")
        return 0

    return numerator / denominator