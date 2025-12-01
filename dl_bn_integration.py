import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

class BookRecommendationSystem:
    """Hybrid Book Recommendation System combining DL and BN"""
    
    def __init__(self, model_path='fitted_bn.pkl'):
        """Initialize the recommendation system"""
        self.model = self.load_bn_model(model_path)
        self.infer = VariableElimination(self.model) if self.model else None
        
    def load_bn_model(self, model_path):
        """Load the fitted BN model"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded BN model from {model_path}")
            return model
        except FileNotFoundError:
            print(f"Error: {model_path} not found. Please run bn_build_and_fit.py first.")
            return None
    
    def get_bn_probability(self, evidence):
        """Get P(Recommendation=Yes | evidence) from BN"""
        if self.model is None:
            return 0.5
        
        try:
            result = self.infer.query(variables=['Recommendation'], evidence=evidence)
            return result.values[0]  # Probability of 'Yes'
        except Exception as e:
            print(f"Error in BN inference: {e}")
            return 0.5
    
    def combine_scores(self, dl_score, bn_prob, alpha=0.7):
        """Combine DL and BN scores"""
        return dl_score * (alpha * bn_prob + (1 - alpha))
    
    def get_user_evidence(self, user_id, user_preferences_df):
        """Get user-specific evidence from preferences"""
        try:
            user_pref = user_preferences_df[user_preferences_df['user_id'] == user_id]['UserPreference'].iloc[0]
            return {
                'UserPreference': user_pref,
                'Sentiment': 'Positive'  # Default, could be derived from user history
            }
        except:
            return {
                'UserPreference': 'CasualReader',  # Default
                'Sentiment': 'Positive'
            }
    
    def recommend_books(self, candidates_df, user_id, user_preferences_df, 
                       alpha=0.7, top_k=10):
        """
        Generate final recommendations by combining DL and BN
        
        Args:
            candidates_df: DataFrame with DL candidates
            user_id: User ID for personalization
            user_preferences_df: DataFrame with user preferences
            alpha: Weight for BN probability
            top_k: Number of top recommendations to return
        
        Returns:
            DataFrame with final recommendations
        """
        print(f"Generating recommendations for user {user_id}")
        
        # Get user evidence
        user_evidence = self.get_user_evidence(user_id, user_preferences_df)
        print(f"User evidence: {user_evidence}")
        
        results = []
        
        for idx, row in candidates_df.iterrows():
            # Prepare evidence for this candidate
            evidence = user_evidence.copy()
            evidence.update({
                'Genre_small': row['genre'],
                'Popularity': row['popularity_bucket'],
                'ReadLength': self.categorize_read_length(row.get('num_pages', 300))
            })
            
            # Get BN probability
            bn_prob = self.get_bn_probability(evidence)
            
            # Combine scores
            dl_score = row['dl_score']
            final_score = self.combine_scores(dl_score, bn_prob, alpha)
            
            results.append({
                'isbn13': row['isbn13'],
                'title': row['title'],
                'genre': row['genre'],
                'num_pages': row.get('num_pages', 0),
                'popularity': row['popularity_bucket'],
                'dl_score': dl_score,
                'bn_probability': bn_prob,
                'final_score': final_score,
                'evidence': evidence
            })
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df.head(top_k)
    
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
    
    def explain_recommendation(self, evidence, top_n=3):
        """Generate explanation for a recommendation"""
        if self.model is None:
            return "BN model not available"
        
        try:
            # Get base probability
            base_prob = self.get_bn_probability(evidence)
            
            explanations = []
            
            # Test each variable's contribution
            for var in ['Sentiment', 'UserPreference', 'Genre_small', 'Popularity', 'ReadLength']:
                if var in evidence:
                    # Get probability without this variable
                    evidence_without = {k: v for k, v in evidence.items() if k != var}
                    if evidence_without:
                        prob_without = self.get_bn_probability(evidence_without)
                        contribution = base_prob - prob_without
                        
                        explanations.append({
                            'variable': var,
                            'value': evidence[var],
                            'contribution': contribution,
                            'abs_contribution': abs(contribution)
                        })
            
            # Sort by absolute contribution
            explanations.sort(key=lambda x: x['abs_contribution'], reverse=True)
            
            # Create explanation text
            explanation_parts = []
            for exp in explanations[:top_n]:
                if exp['contribution'] > 0:
                    direction = "increases"
                else:
                    direction = "decreases"
                
                explanation_parts.append(
                    f"{exp['variable']}={exp['value']} {direction} recommendation probability"
                )
            
            explanation_text = "; ".join(explanation_parts)
            
            return {
                'probability': base_prob,
                'top_contributors': explanations[:top_n],
                'explanation': explanation_text
            }
            
        except Exception as e:
            return f"Error generating explanation: {e}"

def evaluate_recommendations(recommendations_df, test_interactions_df, user_id, top_k=10):
    """Evaluate recommendation quality using test interactions"""
    try:
        # Get user's test interactions
        user_interactions = test_interactions_df[test_interactions_df['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'coverage': 0}
        
        # Get recommended book IDs
        recommended_books = set(recommendations_df['isbn13'].tolist())
        
        # Get user's liked books (rating >= 4)
        liked_books = set(user_interactions[user_interactions['rating'] >= 4]['isbn13'].tolist())
        
        # Calculate metrics
        relevant_recommended = recommended_books.intersection(liked_books)
        
        precision = len(relevant_recommended) / top_k if top_k > 0 else 0
        recall = len(relevant_recommended) / len(liked_books) if len(liked_books) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        coverage = len(recommendations_df) / len(user_interactions) if len(user_interactions) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'coverage': coverage,
            'relevant_recommended': len(relevant_recommended),
            'total_liked': len(liked_books)
        }
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0, 'coverage': 0}

def main():
    """Main function to demonstrate the integrated system"""
    print("Hybrid DL-BN Book Recommendation System")
    print("="*50)
    
    # Initialize the system
    rec_system = BookRecommendationSystem()
    
    if rec_system.model is None:
        print("Cannot proceed without BN model. Please run bn_build_and_fit.py first.")
        return
    
    # Load data
    try:
        candidates_df = pd.read_csv('candidates_df.csv')
        user_preferences_df = pd.read_csv('user_preferences.csv')
        print(f"Loaded {len(candidates_df)} candidates and {len(user_preferences_df)} user preferences")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Test with different users and alpha values
    test_users = [1, 2, 3, 4, 5]
    alpha_values = [0.5, 0.7, 0.9]
    
    all_results = []
    
    for alpha in alpha_values:
        print(f"\nTesting with alpha = {alpha}")
        print("-" * 30)
        
        for user_id in test_users:
            # Generate recommendations
            recommendations = rec_system.recommend_books(
                candidates_df, user_id, user_preferences_df, 
                alpha=alpha, top_k=10
            )
            
            print(f"\nUser {user_id} - Top 5 Recommendations:")
            for idx, row in recommendations.head(5).iterrows():
                print(f"  {row['rank']}. {row['title'][:40]}...")
                print(f"     DL: {row['dl_score']:.3f}, BN: {row['bn_probability']:.3f}, Final: {row['final_score']:.3f}")
            
            # Generate explanation for top recommendation
            if len(recommendations) > 0:
                top_rec = recommendations.iloc[0]
                explanation = rec_system.explain_recommendation(top_rec['evidence'])
                print(f"\n  Explanation for top recommendation:")
                print(f"    {explanation['explanation']}")
            
            # Store results
            all_results.append({
                'alpha': alpha,
                'user_id': user_id,
                'recommendations': recommendations
            })
    
    # Save all results
    final_recommendations = []
    for result in all_results:
        rec_df = result['recommendations'].copy()
        rec_df['alpha'] = result['alpha']
        rec_df['user_id'] = result['user_id']
        final_recommendations.append(rec_df)
    
    if final_recommendations:
        final_df = pd.concat(final_recommendations, ignore_index=True)
        final_df.to_csv('final_recommendations.csv', index=False)
        print(f"\nAll recommendations saved to 'final_recommendations.csv'")
    
    print("\nIntegration testing completed!")

if __name__ == "__main__":
    main()
