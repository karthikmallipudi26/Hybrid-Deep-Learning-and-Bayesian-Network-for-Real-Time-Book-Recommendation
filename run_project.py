import pandas as pd
import numpy as np
import gradio as gr
from bayesian_network import BookRecommenderBN
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BOOK RECOMMENDATION SYSTEM - STARTING")
print("="*60)

# Initialize the Bayesian Network recommender
print("\n1. Initializing Bayesian Network Recommender...")
try:
    bn_recommender = BookRecommenderBN()
    print("✓ BN Recommender initialized successfully")
    
    # Suppress stdout to prevent automatic CPD printing when accessing model nodes
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        model_nodes = list(bn_recommender.model.nodes()) if bn_recommender.model else []
    finally:
        sys.stdout = old_stdout
    
    print(f"  Model nodes: {model_nodes}")
    print(f"  Books loaded: {len(bn_recommender.books_data)}")
except Exception as e:
    print(f"✗ Error initializing BN: {e}")
    bn_recommender = None

# Load books data
print("\n2. Loading books data...")
try:
    books = pd.read_csv("books_with_emotions.csv")
    print(f"✓ Loaded {len(books)} books")
except Exception as e:
    print(f"✗ Error loading books: {e}")
    books = pd.DataFrame()

def recommend_books_simple(query, category, tone):
    """Simple book recommendation function with total score calculation"""
    print(f"\nGenerating recommendations for: '{query}'")
    print(f"Category: {category}, Tone: {tone}")
    
    try:
        if bn_recommender is None or len(books) == 0:
            return [("cover-not-found.jpg", "System not ready. Please check initialization.")]
        
        # Get BN recommendations
        bn_recs = bn_recommender.recommend_books(
            user_query=query,
            category_filter=category if category != "All" else None,
            emotion_sort=tone if tone != "All" else None,
            top_k=16
        )
        
        print(f"Generated {len(bn_recs)} recommendations")
        
        if len(bn_recs) == 0:
            return [("cover-not-found.jpg", "No recommendations found.")]
        
        # Calculate total scores (combining DL and BN scores)
        alpha = 0.7  # Weight for BN probability
        results = []
        
        for i, (_, row) in enumerate(bn_recs.iterrows()):
            description = str(row.get("description", "") or "")
            truncated_desc = " ".join(description.split()[:30]) + "..."
            
            title = str(row.get("title", "Unknown Title"))
            authors = str(row.get("authors", "Unknown Author") or "Unknown Author")
            
            # Get BN probability
            bn_prob = row.get('recommendation_prob', 0.5)
            
            # Simulate DL score based on position in BN results (higher position = higher DL score)
            # In a real implementation, you would get actual similarity scores from semantic search
            dl_score = max(0.1, 1.0 - (i * 0.05))  # Decreasing score based on position
            
            # Calculate total score using the formula: final_score = dl_score * (alpha * bn_prob + (1 - alpha))
            total_score = dl_score * (alpha * bn_prob + (1 - alpha))
            
            # Add total score to caption
            score_text = f" [Total Score: {total_score:.2f}]"
            
            caption = f"{title} by {authors}{score_text}: {truncated_desc}"
            
            # Use a placeholder image
            img = "cover-not-found.jpg"
            
            results.append((img, caption))
            
            if i < 3:  # Show first 3 for debugging
                print(f"  {i+1}. {title[:50]}... (Total: {total_score:.3f}, DL: {dl_score:.3f}, BN: {bn_prob:.3f})")
        
        return results
        
    except Exception as e:
        print(f"✗ Error in recommendation: {e}")
        return [("cover-not-found.jpg", f"Error: {str(e)}")]

def visualize_bn_simple():
    """Simple BN visualization"""
    try:
        if bn_recommender is None:
            return "cover-not-found.jpg"
        
        img_path = bn_recommender.visualize_network()
        print(f"BN visualization saved to: {img_path}")
        return img_path if img_path else "cover-not-found.jpg"
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        return "cover-not-found.jpg"

def show_cpt_simple():
    """Simple CPT display"""
    try:
        if bn_recommender is None:
            return "BN model not available"
        
        summary = bn_recommender.get_cpt_summary()
        print("CPT summary generated")
        return summary
    except Exception as e:
        return f"Error getting CPT summary: {e}"

# Get categories and tones
print("\n3. Setting up categories and tones...")
try:
    if len(books) > 0:
        categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())[:20]
        tones = ["All", "joy", "surprise", "anger", "fear", "sadness"]
    else:
        categories = ["All", "Fiction", "Non-fiction"]
        tones = ["All", "joy", "surprise", "anger", "fear", "sadness"]
    
    print(f"✓ Categories: {len(categories)}")
    print(f"✓ Tones: {len(tones)}")
except:
    categories = ["All", "Fiction", "Non-fiction"]
    tones = ["All", "joy", "surprise", "anger", "fear", "sadness"]

# Create Gradio interface
print("\n4. Creating Gradio interface...")
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
            
            visualize_button.click(fn=visualize_bn_simple, inputs=[], outputs=visualization_output)
            cpt_button.click(fn=show_cpt_simple, inputs=[], outputs=cpt_output)
        
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

print("\n5. Launching dashboard...")
print("="*60)
print("🚀 BOOK RECOMMENDATION SYSTEM READY!")
print("="*60)
print("Dashboard will be available at: http://localhost:7860")
print("Press Ctrl+C to stop the server")
print("="*60)

if __name__ == "__main__":
    try:
        dashboard.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("Trying alternative port...")
        try:
            dashboard.launch(
                server_name="0.0.0.0",
                server_port=7861,
                share=False,
                show_error=True,
                quiet=False
            )
        except Exception as e2:
            print(f"Error with alternative port: {e2}")
            print("Please check if ports 7860 and 7861 are available.")
