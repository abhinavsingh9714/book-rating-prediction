from src.data_loader import load_data, preprocess_ratings, create_sparse_user_item_matrix
from src.data_splitter import split_train_test_sparse
from src.similarity import compute_item_similarity, compute_topk_item_similarity, compute_user_similarity
from src.predictor import predict_rating
from src.evaluation import mean_absolute_difference
import os
from scipy.sparse import save_npz, load_npz
import matplotlib.pyplot as plt
print('-' * 50)
print("Step 1: Data Loading and Preprocessing\n")
books, users, ratings = load_data()
ratings_cleaned = preprocess_ratings(ratings)
print('-' * 50)
print("Step 2: Creating Sparse User-Item Matrix\n")
sparse_matrix, user_encoder, item_encoder = create_sparse_user_item_matrix(ratings_cleaned)
print('-' * 50)
print("Step 3: Splitting Data into Train and Test Sets\n")
train_matrix, test_set = split_train_test_sparse(sparse_matrix, test_ratio=0.25)
print('-' * 50)
print("Step 4: Computing Item Similarity Matrix for different top_k values\n")
# Define the range of top_k values to test
top_k_values = [5, 10, 15, 20, 50, 100]
mae_results_topk = {}

for top_k in top_k_values:
    print(f"Evaluating for top_k={top_k}...")
    # Compute the item similarity matrix for the current top_k
    item_similarity_matrix = compute_topk_item_similarity(train_matrix, top_k=top_k)
    # Evaluate the MAE using the computed similarity matrix
    mad = mean_absolute_difference(test_set, train_matrix, item_similarity_matrix, k=10)
    mae_results_topk[top_k] = mad
    print(f"MAE for top_k={top_k}: {mad:.4f}")

# Print the results for comparison
print("\nMAE Results for different top_k values:")
for top_k, mae in mae_results_topk.items():
    print(f"top_k={top_k}: MAE={mae:.4f}")

kvalue_list = list(mae_results_topk.keys())
mae_results_topk_list = list(mae_results_topk.values())

plt.figure(figsize=(10, 6))
plt.plot(kvalue_list, mae_results_topk_list, marker='o', linestyle='-', color='b')
plt.title('MAE vs Neighborhood size', fontsize=16)
plt.xlabel('Neighborhood size', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.grid(True)
plt.xticks(kvalue_list)
# plt.show()

best_top_k = min(mae_results_topk, key=mae_results_topk.get)
print(f"Best top_k value: {best_top_k} with MAE: {mae_results_topk[best_top_k]:.4f}")
print('-' * 50)
print("Step 5: Evaluating MAE for different train-to-test ratios")
# Define the range of train-to-test ratios to test
train_ratios = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
mae_results_sample_ratio = {}
top_k = best_top_k  # Fixed top_k value

for train_ratio in train_ratios:
    print(f"Evaluating for train_ratio={train_ratio}...")
    # Split the data into train and test sets based on the current train_ratio
    train_matrix, test_set = split_train_test_sparse(sparse_matrix, test_ratio=1 - train_ratio)
    
    # Compute the item similarity matrix for the fixed top_k
    item_similarity_matrix = compute_topk_item_similarity(train_matrix, top_k=top_k)
    
    # Evaluate the MAE using the computed similarity matrix
    mad = mean_absolute_difference(test_set, train_matrix, item_similarity_matrix, k=10)
    mae_results_sample_ratio[train_ratio] = mad
    print(f"MAE for train_ratio={train_ratio}: {mad:.4f}")

# Print the results for comparison
print("\nMAE Results for different train-to-test ratios:")
for train_ratio, mae in mae_results_sample_ratio.items():
    print(f"train_ratio={train_ratio:.2f}: MAE={mae:.4f}")

train_ratios_list = list(mae_results_sample_ratio.keys())
mae_results_sample_ratio_list = list(mae_results_sample_ratio.values())

plt.figure(figsize=(10, 6))
plt.plot(train_ratios_list, mae_results_sample_ratio_list, marker='o', linestyle='-', color='b')
plt.title('MAE vs Train-to-Test Ratios', fontsize=16)
plt.xlabel('Train Ratio', fontsize=14)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
plt.grid(True)
plt.xticks(train_ratios_list)
plt.show()