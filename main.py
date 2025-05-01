from src.data_loader import load_data, preprocess_ratings, create_sparse_user_item_matrix
from src.data_splitter import split_train_test_sparse
from src.similarity import compute_item_similarity, compute_topk_item_similarity
from src.predictor import predict_rating
from src.evaluation import mean_absolute_difference
import os
from scipy.sparse import save_npz, load_npz

books, users, ratings = load_data()
ratings_cleaned = preprocess_ratings(ratings)
sparse_matrix, user_encoder, item_encoder = create_sparse_user_item_matrix(ratings_cleaned)

train_matrix, test_set = split_train_test_sparse(sparse_matrix, test_ratio=0.25)

# train_matrix: (n_users, n_items) from previous step
top_k=5
cache_path=f"results/item_similarity_matrix_{top_k}.npz"
if os.path.exists(cache_path):
    print("📂 Loading cached item similarity matrix...")
    item_similarity_matrix = load_npz(cache_path)
else:
    print("⚙️ Computing item similarity matrix...")
    item_similarity_matrix = compute_topk_item_similarity(train_matrix, top_k=top_k)
    save_npz(cache_path, item_similarity_matrix)

print(f"Item Similarity Matrix shape: {item_similarity_matrix.shape}")
# pred = predict_rating(
#     user_idx=89, 
#     item_idx=17,
#     train_matrix=train_matrix,
#     similarity_matrix=item_similarity_matrix,
#     k=10
# )j
# print(f"Predicted Rating: {pred:.2f}")


mad = mean_absolute_difference(test_set, train_matrix, item_similarity_matrix, k=10)
print(f"Mean Absolute Difference: {mad:.4f}")