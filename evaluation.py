import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Evaluate the hybrid DL-BN recommendation system"""
    
    def __init__(self, model_path='fitted_bn.pkl'):
        """Initialize the evaluator"""
        self.model = self.load_bn_model(model_path)
        self.infer = VariableElimination(self.model) if self.model else None
        
    def load_bn_model(self, model_path):
        """Load the fitted BN model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except FileNotFoundError:
            print(f"Error: {model_path} not found.")
            return None
    
    def get_bn_probability(self, evidence):
        """Get P(Recommendation=Yes | evidence) from BN"""
        if self.model is None:
            return 0.5
        
        try:
            result = self.infer.query(variables=['Recommendation'], evidence=evidence)
            return result.values[0]
        except Exception as e:
            return 0.5
    
    def split_data(self, interactions_df, test_ratio=0.2):
        """Split interactions into train and test sets"""
        # Group by user to ensure each user has both train and test data
        train_data = []
        test_data = []
        
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # Shuffle and split
            shuffled = user_interactions.sample(frac=1).reset_index(drop=True)
            split_idx = int(len(shuffled) * (1 - test_ratio))
            
            train_data.append(shuffled[:split_idx])
            test_data.append(shuffled[split_idx:])
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        return train_df, test_df
    
    def evaluate_precision_at_k(self, recommendations, test_interactions, k=10):
        """Calculate Precision@K"""
        if len(recommendations) == 0:
            return 0
        
        # Get top-k recommendations
        top_k_recs = recommendations.head(k)
        recommended_books = set(top_k_recs['isbn13'].tolist())
        
        # Get relevant books (rating >= 4)
        relevant_books = set(test_interactions[test_interactions['rating'] >= 4]['isbn13'].tolist())
        
        # Calculate precision
        relevant_recommended = recommended_books.intersection(relevant_books)
        precision = len(relevant_recommended) / k if k > 0 else 0
        
        return precision
    
    def evaluate_recall_at_k(self, recommendations, test_interactions, k=10):
        """Calculate Recall@K"""
        if len(recommendations) == 0:
            return 0
        
        # Get top-k recommendations
        top_k_recs = recommendations.head(k)
        recommended_books = set(top_k_recs['isbn13'].tolist())
        
        # Get relevant books (rating >= 4)
        relevant_books = set(test_interactions[test_interactions['rating'] >= 4]['isbn13'].tolist())
        
        if len(relevant_books) == 0:
            return 0
        
        # Calculate recall
        relevant_recommended = recommended_books.intersection(relevant_books)
        recall = len(relevant_recommended) / len(relevant_books)
        
        return recall
    
    def evaluate_mrr(self, recommendations, test_interactions):
        """Calculate Mean Reciprocal Rank"""
        if len(recommendations) == 0:
            return 0
        
        # Get relevant books (rating >= 4)
        relevant_books = set(test_interactions[test_interactions['rating'] >= 4]['isbn13'].tolist())
        
        if len(relevant_books) == 0:
            return 0
        
        # Find rank of first relevant book
        for rank, row in recommendations.iterrows():
            if row['isbn13'] in relevant_books:
                return 1.0 / (rank + 1)
        
        return 0
    
    def evaluate_ndcg_at_k(self, recommendations, test_interactions, k=10):
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if len(recommendations) == 0:
            return 0
        
        # Get top-k recommendations
        top_k_recs = recommendations.head(k)
        
        # Get relevance scores (ratings)
        relevance_scores = []
        for _, row in top_k_recs.iterrows():
            user_ratings = test_interactions[test_interactions['isbn13'] == row['isbn13']]
            if len(user_ratings) > 0:
                relevance_scores.append(user_ratings['rating'].iloc[0])
            else:
                relevance_scores.append(0)
        
        # Calculate DCG
        dcg = 0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return ndcg
    
    def compare_baselines(self, candidates_df, user_preferences_df, test_interactions_df, 
                         alpha_values=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0]):
        """Compare different alpha values and baselines"""
        print("Comparing Different Approaches")
        print("="*40)
        
        results = []
        
        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha}")
            
            user_metrics = []
            
            for user_id in test_interactions_df['user_id'].unique():
                # Get user's test interactions
                user_test = test_interactions_df[test_interactions_df['user_id'] == user_id]
                
                if len(user_test) == 0:
                    continue
                
                # Get user evidence
                try:
                    user_pref = user_preferences_df[user_preferences_df['user_id'] == user_id]['UserPreference'].iloc[0]
                except:
                    user_pref = 'CasualReader'
                
                user_evidence = {
                    'UserPreference': user_pref,
                    'Sentiment': 'Positive'
                }
                
                # Generate recommendations
                recommendations = []
                
                for _, row in candidates_df.iterrows():
                    evidence = user_evidence.copy()
                    evidence.update({
                        'Genre_small': row['genre'],
                        'Popularity': row['popularity_bucket'],
                        'ReadLength': self.categorize_read_length(row.get('num_pages', 300))
                    })
                    
                    bn_prob = self.get_bn_probability(evidence)
                    dl_score = row['dl_score']
                    final_score = dl_score * (alpha * bn_prob + (1 - alpha))
                    
                    recommendations.append({
                        'isbn13': row['isbn13'],
                        'title': row['title'],
                        'final_score': final_score
                    })
                
                # Sort by final score
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df = recommendations_df.sort_values('final_score', ascending=False).reset_index(drop=True)
                
                # Calculate metrics
                precision_10 = self.evaluate_precision_at_k(recommendations_df, user_test, k=10)
                recall_10 = self.evaluate_recall_at_k(recommendations_df, user_test, k=10)
                mrr = self.evaluate_mrr(recommendations_df, user_test)
                ndcg_10 = self.evaluate_ndcg_at_k(recommendations_df, user_test, k=10)
                
                user_metrics.append({
                    'user_id': user_id,
                    'precision_10': precision_10,
                    'recall_10': recall_10,
                    'mrr': mrr,
                    'ndcg_10': ndcg_10
                })
            
            # Calculate average metrics
            if user_metrics:
                avg_metrics = {
                    'alpha': alpha,
                    'precision_10': np.mean([m['precision_10'] for m in user_metrics]),
                    'recall_10': np.mean([m['recall_10'] for m in user_metrics]),
                    'mrr': np.mean([m['mrr'] for m in user_metrics]),
                    'ndcg_10': np.mean([m['ndcg_10'] for m in user_metrics]),
                    'num_users': len(user_metrics)
                }
                results.append(avg_metrics)
                
                print(f"  Precision@10: {avg_metrics['precision_10']:.3f}")
                print(f"  Recall@10: {avg_metrics['recall_10']:.3f}")
                print(f"  MRR: {avg_metrics['mrr']:.3f}")
                print(f"  NDCG@10: {avg_metrics['ndcg_10']:.3f}")
        
        return pd.DataFrame(results)
    
    def categorize_read_length(self, num_pages):
        """Categorize read length based on number of pages"""
        if pd.isna(num_pages) or num_pages == 0:
            return 'unknown'
        elif num_pages < 200:
            return 'short'
        elif num_pages < 400:
            return 'medium'
        else:
            return 'long'
    
    def plot_results(self, results_df):
        """Plot evaluation results"""
        if len(results_df) == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Recommendation System Evaluation Results', fontsize=16)
        
        # Precision@10
        axes[0, 0].plot(results_df['alpha'], results_df['precision_10'], 'b-o')
        axes[0, 0].set_title('Precision@10')
        axes[0, 0].set_xlabel('Alpha (BN Weight)')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].grid(True)
        
        # Recall@10
        axes[0, 1].plot(results_df['alpha'], results_df['recall_10'], 'r-o')
        axes[0, 1].set_title('Recall@10')
        axes[0, 1].set_xlabel('Alpha (BN Weight)')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True)
        
        # MRR
        axes[1, 0].plot(results_df['alpha'], results_df['mrr'], 'g-o')
        axes[1, 0].set_title('Mean Reciprocal Rank (MRR)')
        axes[1, 0].set_xlabel('Alpha (BN Weight)')
        axes[1, 0].set_ylabel('MRR')
        axes[1, 0].grid(True)
        
        # NDCG@10
        axes[1, 1].plot(results_df['alpha'], results_df['ndcg_10'], 'm-o')
        axes[1, 1].set_title('NDCG@10')
        axes[1, 1].set_xlabel('Alpha (BN Weight)')
        axes[1, 1].set_ylabel('NDCG')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main evaluation function"""
    print("Recommendation System Evaluation")
    print("="*40)
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    if evaluator.model is None:
        print("Cannot proceed without BN model. Please run bn_build_and_fit.py first.")
        return
    
    # Load data
    try:
        candidates_df = pd.read_csv('candidates_df.csv')
        user_preferences_df = pd.read_csv('user_preferences.csv')
        
        # Create synthetic test interactions if not available
        try:
            test_interactions_df = pd.read_csv('user_item_interaction.py')
        except:
            print("Creating synthetic test interactions...")
            # Create synthetic test data
            test_data = []
            for user_id in user_preferences_df['user_id'].unique():
                # Sample some books for this user
                user_books = candidates_df.sample(n=min(10, len(candidates_df)))
                for _, book in user_books.iterrows():
                    # Generate rating based on user preference and book popularity
                    base_rating = 3
                    if book['popularity_bucket'] in ['High', 'Very High']:
                        base_rating += 1
                    
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                    rating = max(1, min(5, base_rating + np.random.randint(-1, 2)))
                    
                    test_data.append({
                        'user_id': user_id,
                        'isbn13': book['isbn13'],
                        'rating': rating
                    })
            
            test_interactions_df = pd.DataFrame(test_data)
            test_interactions_df.to_csv('test_interactions.csv', index=False)
            print(f"Created {len(test_interactions_df)} test interactions")
        
        print(f"Loaded {len(candidates_df)} candidates, {len(user_preferences_df)} users, {len(test_interactions_df)} test interactions")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Run evaluation
    results_df = evaluator.compare_baselines(
        candidates_df, user_preferences_df, test_interactions_df,
        alpha_values=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    )
    
    # Save results
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\nEvaluation results saved to 'evaluation_results.csv'")
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 30)
    best_alpha = results_df.loc[results_df['precision_10'].idxmax(), 'alpha']
    best_precision = results_df['precision_10'].max()
    
    print(f"Best alpha value: {best_alpha}")
    print(f"Best Precision@10: {best_precision:.3f}")
    
    # Plot results
    evaluator.plot_results(results_df)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
