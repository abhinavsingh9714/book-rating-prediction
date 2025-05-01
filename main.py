from src.data_loader import load_data, preprocess_ratings, create_sparse_user_item_matrix
from src.data_splitter import split_train_test_sparse
from src.similarity import compute_item_similarity, compute_topk_item_similarity, compute_user_similarity
from src.predictor import predict_rating
from src.evaluation import mean_absolute_difference
import os
from scipy.sparse import save_npz, load_npz

books, users, ratings = load_data()
ratings_cleaned = preprocess_ratings(ratings)
sparse_matrix, user_encoder, item_encoder = create_sparse_user_item_matrix(ratings_cleaned)

train_matrix, test_set = split_train_test_sparse(sparse_matrix, test_ratio=0.25)

# train_matrix: (n_users, n_items) from previous step
# top_k=50
# cache_path=f"results/item_similarity_matrix_{top_k}.npz"
# if os.path.exists(cache_path):
#     print("📂 Loading cached item similarity matrix...")
#     item_similarity_matrix = load_npz(cache_path)
# else:
#     print("⚙️ Computing item similarity matrix...")
#     item_similarity_matrix = compute_topk_item_similarity(train_matrix, top_k=top_k)
#     save_npz(cache_path, item_similarity_matrix)

# print(f"Item Similarity Matrix shape: {item_similarity_matrix.shape}")

# mad = mean_absolute_difference(test_set, train_matrix, item_similarity_matrix, k=10)
# print(f"Mean Absolute Difference: {mad:.4f}")

# ...existing imports...

# Define the range of top_k values to test
top_k_values = [5, 10, 15, 20, 50, 100]
mae_results = {}

for top_k in top_k_values:
    print(f"🔍 Evaluating for top_k={top_k}...")
    # Compute the item similarity matrix for the current top_k
    item_similarity_matrix = compute_topk_item_similarity(train_matrix, top_k=top_k)
    # Evaluate the MAE using the computed similarity matrix
    mad = mean_absolute_difference(test_set, train_matrix, item_similarity_matrix, k=10)
    mae_results[top_k] = mad
    print(f"MAE for top_k={top_k}: {mad:.4f}")

# Print the results for comparison
print("\n📊 MAE Results for different top_k values:")
for top_k, mae in mae_results.items():
    print(f"top_k={top_k}: MAE={mae:.4f}")