import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN warnings
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings

# Suppress ChromaDB telemetry
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

import gradio as gr
from bayesian_network import BookRecommenderBN

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["authors"] = books.get("authors", "").fillna("").astype(str)
books["description"] = books.get("description", "").fillna("").astype(str)
# Create unique placeholder thumbnails for each book using ISBN as seed
books["thumbnail"] = books["isbn13"].apply(lambda x: f"https://picsum.photos/200/300?random={hash(str(x)) % 1000}")
books["large_thumbnail"] = books["isbn13"].apply(lambda x: f"https://picsum.photos/400/600?random={hash(str(x)) % 1000}")

def _build_documents_from_books(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for _, r in df.iterrows():
        isbn = r.get("isbn13")
        if pd.isna(isbn):
            continue
        try:
            isbn = int(isbn)
        except Exception:
            continue
        title = str(r.get("title", "") or "").strip()
        authors_str = str(r.get("authors", "") or "").strip()
        cat = str(r.get("simple_categories", r.get("category", "")) or "").strip()
        desc = str(r.get("description", "") or "").strip()
        if not desc and not title:
            continue
        # Enrich embedding text with title/author/category for better discrimination
        content = " | ".join([p for p in [title, authors_str, cat, desc] if p])
        docs.append(Document(page_content=content, metadata={"isbn13": isbn}))
    return docs

documents: list[Document] = _build_documents_from_books(books)

# Initialize embeddings with fallback if OpenAI key is missing
embeddings = None
openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
use_openai = bool(openai_key) and openai_key.startswith("sk-")
if use_openai:
    try:
        embeddings = OpenAIEmbeddings()
    except Exception:
        embeddings = None
if embeddings is None:
    try:
        # Preferred modern import to avoid deprecation warnings
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        # Fallback to legacy if package not installed
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = Chroma.from_documents(documents, embeddings)

# Initialize Bayesian Network recommender
bn_recommender = BookRecommenderBN(books_data_path="books_with_emotions.csv")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 200,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []
    for rec in recs:
        meta_isbn = rec.metadata.get("isbn13") if hasattr(rec, "metadata") else None
        if meta_isbn is None:
            continue
        try:
            books_list.append(int(meta_isbn))
        except Exception:
            continue
    book_recs = books[books["isbn13"].isin(books_list)].copy()
    # preserve semantic order
    book_recs["_order"] = pd.Categorical(book_recs["isbn13"], categories=books_list, ordered=True)
    book_recs.sort_values("_order", inplace=True)
    book_recs.drop(columns=["_order"], inplace=True)
    book_recs = book_recs.head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    # Get initial recommendations using semantic search
    semantic_recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    # Use Bayesian network to refine recommendations
    # Map tone to emotion for Bayesian network
    emotion_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
        "All": None
    }
    
    # Get Bayesian network recommendations
    bn_recs = bn_recommender.recommend_books(
        user_query=query,
        category_filter=category if category != "All" else None,
        emotion_sort=emotion_map.get(tone),
        top_k=16
    )
    
    # Combine recommendations (prioritize semantic results, then fill with BN)
    combined_recommendations = (
        pd.concat([semantic_recommendations, bn_recs])
        .drop_duplicates(subset=['isbn13'], keep='first')
        .head(16)
    )
    
    results = []
    for _, row in combined_recommendations.iterrows():
        description = row.get("description", "") or ""
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_value = row.get("authors", "")
        authors_value = "" if pd.isna(authors_value) else str(authors_value)
        authors_split = [a for a in authors_value.split(";") if a]
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            # fallback to single string or unknown
            authors_str = authors_value or "Unknown Author"

        # Calculate total score (combining DL and BN scores)
        alpha = 0.7  # Weight for BN probability
        
        # Get BN probability
        bn_prob = float(row.get('recommendation_prob', 0.5)) if 'recommendation_prob' in row and pd.notna(row['recommendation_prob']) else 0.5
        
        # Simulate DL score based on position in results (higher position = higher DL score)
        # In a real implementation, you would get actual similarity scores from semantic search
        dl_score = max(0.1, 1.0 - (len(results) * 0.05))  # Decreasing score based on position
        
        # Calculate total score using the formula: final_score = dl_score * (alpha * bn_prob + (1 - alpha))
        total_score = dl_score * (alpha * bn_prob + (1 - alpha))
        
        # Add total score to caption
        score_text = f" [Total Score: {total_score:.2f}]"
            
        # Resolve a safe image URL/path
        img = row.get("large_thumbnail")
        if pd.isna(img) or not isinstance(img, str) or not img.strip().lower().startswith("http"):
            img = row.get("thumbnail")
        if pd.isna(img) or not isinstance(img, str) or img.strip() == "" or not img.strip().lower().startswith("http"):
            img = "cover-not-found.jpg"

        caption = f"{row['title']} by {authors_str}{score_text}: {truncated_description}"
        results.append((img, caption))
    return results

# Fix categories by handling NaN values
categories = ["All"] + sorted([cat for cat in books["simple_categories"].unique() if pd.notna(cat)])
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

def visualize_bn():
    img_path = bn_recommender.visualize_network()
    return img_path

def get_cpt_tables():
    cpt_summary = bn_recommender.get_cpt_summary()
    return cpt_summary

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender with Bayesian Networks")
    
    with gr.Tabs():
        with gr.TabItem("Book Recommendations"):
            with gr.Row():
                user_query = gr.Textbox(label = "Please enter a description of a book:",
                                    placeholder = "e.g., A story about forgiveness")
                category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
                tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
                submit_button = gr.Button("Find recommendations")

            gr.Markdown("## Recommendations (Enhanced with Bayesian Networks)")
            output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

            submit_button.click(fn = recommend_books,
                            inputs = [user_query, category_dropdown, tone_dropdown],
                            outputs = output)
                            
        with gr.TabItem("Bayesian Network Visualization"):
            gr.Markdown("## Bayesian Network Model")
            gr.Markdown("This tab shows the structure and parameters of the Bayesian network used for book recommendations.")
            
            with gr.Row():
                with gr.Column():
                    visualize_button = gr.Button("Visualize Bayesian Network")
                    visualization_output = gr.Image(label="BN Graph", type="filepath")
                
                with gr.Column():
                    cpt_button = gr.Button("Show CPT Tables")
                    cpt_output = gr.Markdown(label="Conditional Probability Tables")
            
            visualize_button.click(fn=visualize_bn, inputs=[], outputs=visualization_output)
            cpt_button.click(fn=get_cpt_tables, inputs=[], outputs=cpt_output)


if __name__ == "__main__":
    dashboard.launch()