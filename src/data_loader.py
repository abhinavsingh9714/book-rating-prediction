import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder



def load_data(data_dir="./data"):
    """
    Efficiently load large Books, Users, and Ratings datasets.
    Returns:
        books_df, users_df, ratings_df
    """
    print("Loading datasets...")

    books_path = os.path.join(data_dir, "Books.csv")
    users_path = os.path.join(data_dir, "Users.csv")
    ratings_path = os.path.join(data_dir, "Ratings.csv")

    books_df = pd.read_csv(
        books_path,
        usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"],
        encoding='latin-1',
        low_memory=False
    )

    users_df = pd.read_csv(
        users_path,
        usecols=["User-ID", "Location", "Age"],
        encoding='latin-1',
        low_memory=False,
        dtype={"User-ID": int, "Location": str, "Age": "float32"}
    )

    ratings_df = pd.read_csv(
        ratings_path,
        usecols=["User-ID", "ISBN", "Book-Rating"],
        encoding='latin-1',
        low_memory=False,
        dtype={"User-ID": int, "ISBN": str, "Book-Rating": int}
    )

    print("Loaded books:", books_df.shape)
    print("Loaded users:", users_df.shape)
    print("Loaded ratings:", ratings_df.shape)
    return books_df, users_df, ratings_df


def preprocess_ratings(ratings_df):
    """
    Keep only explicit ratings (rating > 0)
    """
    cleaned = ratings_df[ratings_df['Book-Rating'] > 0].copy()
    print(f"Filtered explicit ratings: {cleaned.shape}")
    return cleaned


def create_user_item_matrix(cleaned_ratings_df):
    """
    Create sparse user-item matrix (users as rows, ISBNs as columns)
    """
    print("Creating user-item matrix...")
    matrix = cleaned_ratings_df.pivot_table(
        index='User-ID',
        columns='ISBN',
        values='Book-Rating'
    )
    print("Matrix shape:", matrix.shape)
    return matrix

def create_sparse_user_item_matrix(cleaned_ratings_df):
    """
    Create a sparse user-item matrix with LabelEncoded user and book indices.
    Returns:
        sparse_matrix, user_encoder, item_encoder
    """
    print("Creating sparse user-item matrix...")

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_ids = user_encoder.fit_transform(cleaned_ratings_df["User-ID"])
    item_ids = item_encoder.fit_transform(cleaned_ratings_df["ISBN"])

    sparse_matrix = csr_matrix(
        (cleaned_ratings_df["Book-Rating"], (user_ids, item_ids)),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )

    print(f"Sparse matrix shape: {sparse_matrix.shape} | nnz: {sparse_matrix.nnz} ratings")
    return sparse_matrix, user_encoder, item_encoder

# if __name__ == "__main__":
#     books, users, ratings = load_data()
#     ratings_cleaned = preprocess_ratings(ratings)
#     # matrix = create_user_item_matrix(ratings_cleaned)

#     # print("🎯 Data loading and matrix construction complete.")
#     sparse_matrix, user_encoder, item_encoder = create_sparse_user_item_matrix(ratings_cleaned)

#     print("🎯 Sparse matrix created and ready for similarity computations.")