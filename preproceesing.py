import kagglehub
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download latest version of the Kaggle dataset
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
print("Path to dataset files:", path)

# Load the dataset
books = pd.read_csv(f"{path}/books.csv")
print("Loaded books DataFrame shape:", books.shape)

# Visualize missing values
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
plt.title("Missing Values Heatmap")
plt.show()

# Add missing description flag and age of book
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - pd.to_numeric(books["published_year"], errors="coerce")

# Correlation heatmap for columns of interest
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = books[columns_of_interest].corr(method="spearman")
sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                      cbar_kws={"label": "Spearman correlation"})
heatmap.set_title("Correlation Heatmap")
plt.show()

# Filter out rows with missing critical fields
book_missing = books[~(books["description"].isna()) &
                     ~(books["num_pages"].isna()) &
                     ~(books["average_rating"].isna()) &
                     ~(books["published_year"].isna())]
print("Filtered DataFrame shape (book_missing):", book_missing.shape)

# Count categories
print("Category counts:")
print(book_missing["categories"].value_counts().reset_index().sort_values("count", ascending=False))

# Add word count for descriptions
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()

# Update descriptions based on word count ranges
book_missing.loc[book_missing["words_in_description"].between(1, 4), "description"] = "Short description"
book_missing.loc[book_missing["words_in_description"].between(5, 14), "description"] = "Medium description"
book_missing.loc[book_missing["words_in_description"].between(15, 24), "description"] = "Long description"

# Filter to descriptions with 25+ words
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]
print("Filtered DataFrame shape (25+ words):", book_missing_25_words.shape)

# Combine title and subtitle
book_missing_25_words["title_and_subtitle"] = (
    np.where(book_missing_25_words["subtitle"].isna(),
             book_missing_25_words["title"],
             book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
)

# Add tagged description (ISBN + title + subtitle + description)
book_missing_25_words["tagged_description"] = (
    book_missing_25_words[["isbn13", "title_and_subtitle", "description"]].astype(str).agg(" ".join, axis=1)
)

# Save cleaned raw dataset
df = book_missing_25_words.drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
df.to_csv("books_cleaned_raw.csv", index=False)
print("Saved raw cleaned CSV -> books_cleaned_raw.csv")

# ---- BN-friendly feature engineering ----
# Work on a copy of the cleaned DataFrame
df_proc = df.copy()

# Ensure canonical ID column
if "isbn13" not in df_proc.columns:
    if "isbn" in df_proc.columns:
        df_proc["isbn13"] = df_proc["isbn"].astype(str)
    elif "ISBN" in df_proc.columns:
        df_proc["isbn13"] = df_proc["ISBN"].astype(str)
    else:
        df_proc["isbn13"] = df_proc.index.astype(str)

# Age of book
if "published_year" in df_proc.columns:
    df_proc["age_of_book"] = 2024 - pd.to_numeric(df_proc["published_year"], errors="coerce")
else:
    df_proc["age_of_book"] = np.nan

# Read length categories (Short / Medium / Long)
df_proc["num_pages"] = pd.to_numeric(df_proc.get("num_pages", None), errors='coerce')
df_proc["read_length_cat"] = pd.cut(df_proc["num_pages"].fillna(-1),
                                    bins=[-1, 100, 300, 600, 1e6],
                                    labels=["Unknown", "Short", "Medium", "Long"])

# Popularity: prefer ratings_count-like if present, else use average_rating buckets
ratings_cols = [c for c in df_proc.columns if "rating" in c.lower() and "count" in c.lower()]
if ratings_cols:
    col = ratings_cols[0]
    df_proc["popularity_bucket"] = pd.cut(df_proc[col].fillna(0),
                                          bins=[-1, 5, 20, 100, 1e9],
                                          labels=["Low", "Medium", "High", "Very High"])
else:
    # Fallback to average_rating (ensured to exist)
    df_proc["popularity_bucket"] = pd.cut(df_proc["average_rating"].fillna(0),
                                          bins=[-1, 2.5, 3.5, 4.2, 5.0],
                                          labels=["Low", "Medium", "High", "Very High"])

# Ensure genre column (rename categories to genre)
if "genre" not in df_proc.columns:
    if "categories" in df_proc.columns:
        df_proc["genre"] = df_proc["categories"].copy()
    else:
        df_proc["genre"] = "Unknown"

# Sentiment placeholder (will be filled by sentiment analysis)
if "dominant_emotion" not in df_proc.columns:
    df_proc["dominant_emotion"] = np.nan

# Trim down to columns needed for BN, including description and tagged_description
bn_columns = ["isbn13", "title", "genre", "num_pages", "read_length_cat",
              "popularity_bucket", "average_rating", "age_of_book", "dominant_emotion",
              "description", "tagged_description"]
# Keep only those that exist
bn_columns = [c for c in bn_columns if c in df_proc.columns]
bn_df = df_proc[bn_columns].copy()

print("BN-ready columns:", bn_df.columns.tolist())
print("BN-ready shape:", bn_df.shape)

# Save BN-ready CSV
bn_df.to_csv("books_cleaned_bn.csv", index=False)
print("Saved BN-ready CSV -> books_cleaned_bn.csv")

# ---- Diagnostics and next-step hints ----
print("Diagnostics:")
print("  Raw cleaned rows:", len(df))
print("  BN-ready rows:", len(bn_df))
print("\nNext steps (add these cells to this notebook):")
print("  1) Sentiment analysis -> fills 'dominant_emotion' in books_cleaned_bn.csv")
print("  2) Vector search -> generates embeddings and saves embeddings_by_isbn.pkl")
print("  3) Create user_item_interactions.csv if you have ratings or logs")
print("  4) Cluster user preferences -> creates user_preferences.csv")
print("  5) BN build and fit -> creates fitted_bn.pkl and shows CPT tables")
print("  6) BN inference -> tests inference and shows probabilities/explanations")
print("\nFiles produced:")
print(" - books_cleaned_raw.csv")
print(" - books_cleaned_bn.csv")