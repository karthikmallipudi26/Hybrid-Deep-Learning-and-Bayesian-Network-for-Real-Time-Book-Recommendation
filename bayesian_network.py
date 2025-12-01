import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.inference import VariableElimination
import warnings
import sys
from io import StringIO
warnings.filterwarnings('ignore')

def suppress_cpd_output(func):
    """Decorator to suppress CPD printing from pgmpy"""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            sys.stdout = old_stdout
    return wrapper

class SuppressCPDOutput:
    """Context manager to suppress CPD printing from pgmpy"""
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

class BookRecommenderBN:
    """Bayesian Network Book Recommender"""
    
    def __init__(self, books_data_path="books_with_emotions.csv", model_path="fitted_bn.pkl", suppress_cpd_output=True):
        """Initialize the Bayesian Network recommender"""
        self.books_data_path = books_data_path
        self.model_path = model_path
        self.model = None
        self.books_data = None
        self.infer = None
        self.suppress_cpd_output = suppress_cpd_output
        
        # Load the model and data
        self._load_model()
        self._load_books_data()
    
    @suppress_cpd_output
    def _load_model(self):
        """Load the fitted Bayesian Network model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.infer = VariableElimination(self.model)
            print("✓ BN model loaded successfully")
        except FileNotFoundError:
            print(f"Warning: {self.model_path} not found. Using fallback model.")
            self._create_fallback_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._create_fallback_model()
    
    @suppress_cpd_output
    def _create_fallback_model(self):
        """Create a simple fallback model if the main model is not available"""
        print("Creating fallback BN model...")
        
        try:
            from pgmpy.models import DiscreteBayesianNetwork
            from pgmpy.estimators import BayesianEstimator
            
            # Simple structure
            edges = [
                ('Sentiment', 'Recommendation'),
                ('UserPreference', 'Recommendation'),
                ('Genre_small', 'Recommendation')
            ]
            
            model = DiscreteBayesianNetwork(edges)
            
            # Create simple training data
            data = []
            for i in range(100):
                sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.4, 0.4, 0.2])
                user_pref = np.random.choice(['GenreFan', 'CasualReader'], p=[0.5, 0.5])
                genre = np.random.choice(['Fiction', 'Non-fiction'], p=[0.6, 0.4])
                
                prob_yes = 0.3
                if sentiment == 'Positive':
                    prob_yes += 0.3
                if user_pref == 'GenreFan' and genre == 'Fiction':
                    prob_yes += 0.2
                
                recommendation = 'Yes' if np.random.random() < prob_yes else 'No'
                
                data.append({
                    'Sentiment': sentiment,
                    'UserPreference': user_pref,
                    'Genre_small': genre,
                    'Recommendation': recommendation
                })
            
            df = pd.DataFrame(data)
            estimator = BayesianEstimator(model, df)
            fitted_model = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=10)
            model.add_cpds(*fitted_model)
            
            self.model = model
            self.infer = VariableElimination(model)
            print("✓ Fallback model created")
        except Exception as e:
            print(f"Error creating fallback model: {e}")
            # Create a minimal model
            self.model = None
            self.infer = None
    
    def _load_books_data(self):
        """Load books data with emotions"""
        try:
            self.books_data = pd.read_csv(self.books_data_path)
            print(f"✓ Books data loaded: {len(self.books_data)} books")
        except FileNotFoundError:
            print(f"Warning: {self.books_data_path} not found. Using fallback data.")
            self._create_fallback_data()
        except Exception as e:
            print(f"Error loading books data: {e}")
            self._create_fallback_data()
    
    def _create_fallback_data(self):
        """Create fallback books data if the main file is not available"""
        print("Creating fallback books data...")
        
        # Use existing data if available
        try:
            books_df = pd.read_csv('books_cleaned_bn.csv')
            # Add emotion columns if not present
            if 'joy' not in books_df.columns:
                books_df['joy'] = np.random.uniform(0, 1, len(books_df))
                books_df['surprise'] = np.random.uniform(0, 1, len(books_df))
                books_df['anger'] = np.random.uniform(0, 1, len(books_df))
                books_df['fear'] = np.random.uniform(0, 1, len(books_df))
                books_df['sadness'] = np.random.uniform(0, 1, len(books_df))
                books_df['neutral'] = np.random.uniform(0, 1, len(books_df))
            
            # Map columns to expected format
            if 'simple_categories' not in books_df.columns and 'genre' in books_df.columns:
                books_df['simple_categories'] = books_df['genre']
            
            self.books_data = books_df
            print(f"✓ Fallback data created: {len(self.books_data)} books")
            
        except Exception as e:
            print(f"Error creating fallback data: {e}")
            # Create minimal data
            self.books_data = pd.DataFrame({
                'isbn13': [9780000000001, 9780000000002, 9780000000003],
                'title': ['Sample Book 1', 'Sample Book 2', 'Sample Book 3'],
                'authors': ['Author 1', 'Author 2', 'Author 3'],
                'description': ['Description 1', 'Description 2', 'Description 3'],
                'simple_categories': ['Fiction', 'Non-fiction', 'Fiction'],
                'thumbnail': ['', '', ''],
                'joy': [0.5, 0.3, 0.7],
                'surprise': [0.2, 0.4, 0.1],
                'anger': [0.1, 0.2, 0.1],
                'fear': [0.3, 0.1, 0.2],
                'sadness': [0.2, 0.3, 0.1],
                'neutral': [0.4, 0.5, 0.3]
            })
    
    def map_genre_to_bn_category(self, genre):
        """Map any genre to a valid BN category from the actual model states"""
        if pd.isna(genre) or genre == '' or genre == 'nan':
            return 'other'
        
        genre_str = str(genre).lower()
        
        # Map to actual BN model states (from the model we just checked)
        if any(word in genre_str for word in ['fiction', 'novel', 'story', 'tale']):
            return 'Fiction'
        elif any(word in genre_str for word in ['biography', 'autobiography', 'memoir', 'life']):
            return 'Biography & Autobiography'
        elif any(word in genre_str for word in ['science', 'technology', 'computer', 'medical']):
            return 'Science'
        elif any(word in genre_str for word in ['history', 'historical', 'war', 'battle']):
            return 'History'
        elif any(word in genre_str for word in ['art', 'design', 'photography', 'creative']):
            return 'Art'
        elif any(word in genre_str for word in ['religion', 'christian', 'spiritual', 'theology']):
            return 'Religion'
        elif any(word in genre_str for word in ['philosophy', 'psychology', 'self-help', 'mind']):
            return 'Philosophy'
        elif any(word in genre_str for word in ['travel', 'adventure', 'exploration']):
            return 'Travel'
        elif any(word in genre_str for word in ['cooking', 'food', 'recipe', 'culinary']):
            return 'Cooking'
        elif any(word in genre_str for word in ['business', 'economics', 'finance', 'management']):
            return 'Business & Economics'
        elif any(word in genre_str for word in ['health', 'fitness', 'medical', 'wellness']):
            return 'Health & Fitness'
        elif any(word in genre_str for word in ['education', 'learning', 'teaching', 'academic']):
            return 'Education'
        elif any(word in genre_str for word in ['law', 'legal', 'court', 'justice']):
            return 'Law'
        elif any(word in genre_str for word in ['music', 'musical', 'song', 'concert']):
            return 'Music'
        elif any(word in genre_str for word in ['poetry', 'poem', 'verse', 'lyric']):
            return 'Poetry'
        elif any(word in genre_str for word in ['drama', 'play', 'theater', 'theatre']):
            return 'Drama'
        elif any(word in genre_str for word in ['comics', 'graphic', 'manga', 'cartoon']):
            return 'Comics & Graphic Novels'
        elif any(word in genre_str for word in ['juvenile', 'children', 'kids', 'young']):
            return 'Juvenile Fiction'
        elif any(word in genre_str for word in ['fantasy', 'magic', 'wizard', 'dragon']):
            return 'Fantasy fiction'
        elif any(word in genre_str for word in ['mystery', 'detective', 'crime', 'thriller']):
            return 'Detective and mystery stories'
        else:
            return 'other'
    
    @suppress_cpd_output
    def get_bn_probability(self, evidence):
        """Get P(Recommendation=Yes | evidence) from BN"""
        if self.model is None:
            return 0.5
        
        try:
            # Map genre to valid BN category
            if 'Genre_small' in evidence:
                evidence['Genre_small'] = self.map_genre_to_bn_category(evidence['Genre_small'])
            
            result = self.infer.query(variables=['Recommendation'], evidence=evidence)
            prob = result.values[0]  # Probability of 'Yes'
            
            # Ensure we return a valid probability
            if pd.isna(prob) or prob is None:
                return 0.5
            return float(prob)
        except Exception as e:
            print(f"Error in BN inference: {e}")
            return 0.5
    
    @suppress_cpd_output
    def recommend_books(self, user_query, category_filter=None, emotion_sort=None, top_k=16):
        """Generate book recommendations using Bayesian Network"""
        if self.books_data is None:
            return pd.DataFrame()
        
        # Start with all books
        recommendations = self.books_data.copy()
        
        # Apply category filter
        if category_filter and category_filter != "All":
            recommendations = recommendations[recommendations['simple_categories'] == category_filter]
        
        # Calculate BN probabilities for each book
        bn_scores = []
        for _, row in recommendations.iterrows():
            # Map book features to BN evidence using actual model states
            evidence = {
                'Sentiment': 'Positive',  # Default
                'UserPreference': 'GenreFan',  # Default
                'Genre_small': self.map_genre_to_bn_category(row.get('simple_categories', 'Fiction')),
                'Popularity': 'High',  # Default
                'ReadLength': 'medium'  # Default
            }
            
            # Ensure all evidence values are valid states
            valid_states = {
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'UserPreference': ['AuthorLoyal', 'CasualReader', 'DiverseReader', 'GenreFan'],
                'Popularity': ['High', 'Low', 'Medium', 'Very High'],
                'ReadLength': ['long', 'medium', 'short', 'unknown']
            }
            
            # Validate and fix evidence
            for var, states in valid_states.items():
                if evidence[var] not in states:
                    evidence[var] = states[0]  # Use first valid state as fallback
            
            # Get BN probability
            bn_prob = self.get_bn_probability(evidence)
            bn_scores.append(bn_prob)
        
        recommendations['recommendation_prob'] = bn_scores
        
        # Sort by emotion if specified
        if emotion_sort:
            if emotion_sort in recommendations.columns:
                recommendations = recommendations.sort_values(by=emotion_sort, ascending=False)
            else:
                # Fallback to BN probability
                recommendations = recommendations.sort_values(by='recommendation_prob', ascending=False)
        else:
            # Sort by BN probability
            recommendations = recommendations.sort_values(by='recommendation_prob', ascending=False)
        
        return recommendations.head(top_k)
    
    def get_cpt_tables(self):
        """Get Conditional Probability Tables for all variables"""
        if self.model is None:
            return {}
        
        cpt_tables = {}
        try:
            # Suppress stdout to prevent automatic CPD printing
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            for node in self.model.nodes():
                if hasattr(self.model, 'get_cpds'):
                    cpd = self.model.get_cpds(node)
                    if cpd is not None:
                        cpt_tables[node] = cpd
            
            # Restore stdout
            sys.stdout = old_stdout
            return cpt_tables
        except Exception as e:
            # Restore stdout in case of error
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            print(f"Error getting CPT tables: {e}")
            return {}
    
    def visualize_network(self, save_path="bn_visualization.png"):
        """Visualize the Bayesian Network structure"""
        if self.model is None:
            return None
        
        try:
            # Create network graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in self.model.nodes():
                G.add_node(node)
            
            # Add edges
            for edge in self.model.edges():
                G.add_edge(edge[0], edge[1])
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=2000, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title("Bayesian Network Structure for Book Recommendations", 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def get_cpt_summary(self):
        """Get a simplified summary of all Conditional Probability Tables"""
        if self.model is None:
            return "No model loaded"
        
        try:
            # Suppress stdout to prevent automatic CPD printing
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            summary = "## Conditional Probability Tables (CPTs)\n\n"
            cpt_tables = self.get_cpt_tables()
            
            for node, cpd in cpt_tables.items():
                summary += f"### {node}\n"
                summary += f"**States:** {list(cpd.state_names[node])}\n"
                
                # Get parent variables (excluding the node itself)
                parents = [var for var in cpd.variables if var != node]
                if parents:
                    summary += f"**Parents:** {parents}\n"
                    for parent in parents:
                        summary += f"  - {parent}: {list(cpd.state_names[parent])}\n"
                else:
                    summary += "**Parents:** None (Root node)\n"
                
                summary += f"**Shape:** {cpd.values.shape}\n"
                
                # Show probability values in a simpler format
                summary += "**Probability Values:**\n"
                summary += "```\n"
                
                # Convert numpy array to list and format
                try:
                    if len(cpd.values.shape) == 1:
                        # Root node - simple array
                        values_list = cpd.values.tolist()
                        for i, state in enumerate(cpd.state_names[node]):
                            summary += f"  {state}: {values_list[i]:.4f}\n"
                    else:
                        # Child node - show first few combinations
                        values_list = cpd.values.tolist()
                        summary += f"  Array shape: {cpd.values.shape}\n"
                        summary += f"  Total combinations: {len(values_list[0]) if len(values_list) > 0 else 0}\n"
                        summary += f"  Sample values: {values_list[0][:3] if len(values_list) > 0 and len(values_list[0]) > 0 else 'N/A'}\n"
                except Exception as e:
                    summary += f"  Error displaying values: {str(e)}\n"
                
                summary += "```\n\n"
            
            # Restore stdout
            sys.stdout = old_stdout
            return summary
        except Exception as e:
            # Restore stdout in case of error
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            return f"Error generating CPT summary: {e}"
