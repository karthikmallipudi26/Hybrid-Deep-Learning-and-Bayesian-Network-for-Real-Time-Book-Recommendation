import pandas as pd
import numpy as np
import gradio as gr
from bayesian_network import BookRecommenderBN
import warnings
warnings.filterwarnings('ignore')

# Initialize the Bayesian Network recommender
print("Initializing Bayesian Network Recommender...")
bn_recommender = BookRecommenderBN()

# Load books data
books = pd.read_csv("books_with_emotions.csv")
print(f"Loaded {len(books)} books")

def recommend_books_simple(query, category, tone):
    """Simple book recommendation function"""
    try:
        # Get BN recommendations
        bn_recs = bn_recommender.recommend_books(
            user_query=query,
            category_filter=category if category != "All" else None,
            emotion_sort=tone if tone != "All" else None,
            top_k=16
        )
        
        results = []
        for _, row in bn_recs.iterrows():
            description = str(row.get("description", "") or "")
            truncated_desc = " ".join(description.split()[:30]) + "..."
            
            title = str(row.get("title", "Unknown Title"))
            authors = str(row.get("authors", "Unknown Author") or "Unknown Author")
            
            # Add BN probability to caption
            bn_prob = f" [BN Score: {row['recommendation_prob']:.2f}]" if 'recommendation_prob' in row else ""
            
            caption = f"{title} by {authors}{bn_prob}: {truncated_desc}"
            
            # Use a placeholder image
            img = "cover-not-found.jpg"
            
            results.append((img, caption))
        
        return results
        
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return [("cover-not-found.jpg", f"Error: {str(e)}")]

def visualize_bn():
    """Visualize the Bayesian Network"""
    try:
        img_path = bn_recommender.visualize_network()
        return img_path if img_path else "cover-not-found.jpg"
    except Exception as e:
        print(f"Error in visualization: {e}")
        return "cover-not-found.jpg"

def show_cpt_summary():
    """Show CPT summary"""
    try:
        summary = bn_recommender.get_cpt_summary()
        return summary
    except Exception as e:
        return f"Error getting CPT summary: {e}"

# Get categories and tones
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "joy", "surprise", "anger", "fear", "sadness"]

print(f"Categories: {len(categories)}")
print(f"Tones: {tones}")

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Glass(), title="Book Recommendation System") as dashboard:
    gr.Markdown("# 📚 Hybrid DL-BN Book Recommendation System")
    gr.Markdown("This system combines Deep Learning semantic search with Bayesian Network personalization for book recommendations.")
    
    with gr.Tabs():
        with gr.TabItem("🔍 Book Recommendations"):
            gr.Markdown("## Get Personalized Book Recommendations")
            
            with gr.Row():
                user_query = gr.Textbox(
                    label="Describe what kind of book you're looking for:",
                    placeholder="e.g., A story about friendship and adventure",
                    lines=2
                )
                
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    label="Select a category:",
                    value="All"
                )
                tone_dropdown = gr.Dropdown(
                    choices=tones,
                    label="Select an emotional tone:",
                    value="All"
                )
                
            submit_button = gr.Button("🔍 Find Recommendations", variant="primary")
            
            gr.Markdown("## 📖 Recommended Books (Enhanced with Bayesian Networks)")
            output = gr.Gallery(
                label="Recommended books",
                columns=4,
                rows=4,
                height="auto"
            )
            
            submit_button.click(
                fn=recommend_books_simple,
                inputs=[user_query, category_dropdown, tone_dropdown],
                outputs=output
            )
        
        with gr.TabItem("🧠 Bayesian Network Visualization"):
            gr.Markdown("## Bayesian Network Model Structure")
            gr.Markdown("This shows the structure of the Bayesian Network used for personalized recommendations.")
            
            with gr.Row():
                visualize_button = gr.Button("📊 Visualize Bayesian Network", variant="secondary")
                cpt_button = gr.Button("📋 Show CPT Summary", variant="secondary")
            
            with gr.Row():
                visualization_output = gr.Image(
                    label="BN Graph",
                    type="filepath",
                    height=400
                )
                cpt_output = gr.Textbox(
                    label="Conditional Probability Tables (CPTs)",
                    lines=20,
                    max_lines=30
                )
            
            visualize_button.click(fn=visualize_bn, inputs=[], outputs=visualization_output)
            cpt_button.click(fn=show_cpt_summary, inputs=[], outputs=cpt_output)
        
        with gr.TabItem("ℹ️ System Information"):
            gr.Markdown("## System Overview")
            gr.Markdown("""
            ### 🏗️ Architecture
            - **Deep Learning Component**: Semantic similarity search using transformer embeddings
            - **Bayesian Network Component**: Probabilistic reasoning for personalization
            - **Hybrid Integration**: Combines DL scores with BN probabilities
            
            ### 🧠 Bayesian Network Features
            - **6 Nodes**: Sentiment, UserPreference, Genre_small, Popularity, ReadLength, Recommendation
            - **9 Edges**: Realistic dependencies between variables
            - **CPTs**: Conditional Probability Tables learned from data
            
            ### 📊 Key Metrics
            - **Books in Database**: 5,197 books
            - **Categories**: Multiple genres available
            - **Emotions**: 6 emotional dimensions (joy, surprise, anger, fear, sadness, neutral)
            - **BN Probabilities**: Range from 0.3 to 0.9 for realistic recommendations
            
            ### 🔧 Technical Details
            - **Model**: DiscreteBayesianNetwork with BDeu prior
            - **Inference**: Variable Elimination
            - **Training Data**: 1,540 examples (540 real + 1,000 synthetic)
            - **Integration**: Final Score = DL_Score × (α × BN_Probability + (1-α))
            """)

if __name__ == "__main__":
    print("Starting Gradio Dashboard...")
    print("The dashboard will be available at: http://localhost:7860")
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
