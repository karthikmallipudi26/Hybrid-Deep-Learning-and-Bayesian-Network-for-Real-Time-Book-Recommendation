import pandas as pd
import numpy as np
import pickle
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

def load_fitted_model():
    """Load the fitted BN model"""
    try:
        with open('fitted_bn.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Loaded fitted BN model successfully")
        return model
    except FileNotFoundError:
        print("Error: fitted_bn.pkl not found. Please run bn_build_and_fit.py first.")
        return None

def infer_recommendation_prob(model, evidence):
    """
    Infer P(Recommendation=Yes | evidence)
    
    Args:
        model: Fitted Bayesian Network
        evidence: Dictionary of evidence variables and their values
    
    Returns:
        float: Probability of recommendation
    """
    try:
        # Create inference object
        infer = VariableElimination(model)
        
        # Query the probability
        result = infer.query(variables=['Recommendation'], evidence=evidence)
        
        # Extract probability of 'Yes'
        prob_yes = result.values[0]  # First value is 'Yes', second is 'No'
        
        return prob_yes
    except Exception as e:
        print(f"Error in inference: {e}")
        return 0.5  # Default probability

def test_inference_examples(model):
    """Test BN inference with various evidence combinations"""
    print("\n" + "="*60)
    print("TESTING BN INFERENCE")
    print("="*60)
    
    # Test cases with different evidence combinations
    test_cases = [
        {
            'name': 'Positive sentiment, GenreFan, High popularity',
            'evidence': {
                'Sentiment': 'Positive',
                'UserPreference': 'GenreFan',
                'Genre_small': 'Fiction',
                'Popularity': 'High',
                'ReadLength': 'medium'
            }
        },
        {
            'name': 'Neutral sentiment, CasualReader, Low popularity',
            'evidence': {
                'Sentiment': 'Neutral',
                'UserPreference': 'CasualReader',
                'Genre_small': 'other',
                'Popularity': 'Low',
                'ReadLength': 'short'
            }
        },
        {
            'name': 'Negative sentiment, AuthorLoyal, Very High popularity',
            'evidence': {
                'Sentiment': 'Negative',
                'UserPreference': 'AuthorLoyal',
                'Genre_small': 'Fiction',
                'Popularity': 'Very High',
                'ReadLength': 'long'
            }
        },
        {
            'name': 'Positive sentiment, DiverseReader, Medium popularity',
            'evidence': {
                'Sentiment': 'Positive',
                'UserPreference': 'DiverseReader',
                'Genre_small': 'Biography & Autobiography',
                'Popularity': 'Medium',
                'ReadLength': 'medium'
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 50)
        
        # Print evidence
        for var, value in test_case['evidence'].items():
            print(f"{var}: {value}")
        
        # Infer probability
        prob = infer_recommendation_prob(model, test_case['evidence'])
        
        print(f"\nP(Recommendation=Yes | evidence) = {prob:.4f}")
        
        results.append({
            'test_case': test_case['name'],
            'evidence': test_case['evidence'],
            'probability': prob
        })
        
        # Interpretation
        if prob > 0.7:
            interpretation = "Strong recommendation"
        elif prob > 0.5:
            interpretation = "Moderate recommendation"
        else:
            interpretation = "Weak recommendation"
        
        print(f"Interpretation: {interpretation}")
    
    return results

def rerank_candidates(model, candidates_df, user_evidence, alpha=0.7):
    """
    Re-rank candidates using BN probabilities
    
    Args:
        model: Fitted BN model
        candidates_df: DataFrame with candidates and their DL scores
        user_evidence: Dictionary with user-specific evidence
        alpha: Weight for BN probability (0-1)
    
    Returns:
        DataFrame with final scores and rankings
    """
    print(f"\nRe-ranking {len(candidates_df)} candidates with alpha={alpha}")
    
    results = []
    
    for idx, row in candidates_df.iterrows():
        # Prepare evidence for this candidate
        evidence = user_evidence.copy()
        evidence.update({
            'Genre_small': row['genre'],
            'Popularity': row['popularity_bucket'],
            'ReadLength': 'medium'  # Default, could be derived from num_pages
        })
        
        # Get BN probability
        bn_prob = infer_recommendation_prob(model, evidence)
        
        # Calculate final score
        dl_score = row['dl_score']
        final_score = dl_score * (alpha * bn_prob + (1 - alpha))
        
        results.append({
            'isbn13': row['isbn13'],
            'title': row['title'],
            'dl_score': dl_score,
            'bn_probability': bn_prob,
            'final_score': final_score,
            'genre': row['genre'],
            'popularity': row['popularity_bucket']
        })
    
    # Convert to DataFrame and sort by final score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_score', ascending=False).reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    return results_df

def explain_recommendation(model, evidence, top_n=3):
    """
    Generate explanation for a recommendation
    
    Args:
        model: Fitted BN model
        evidence: Evidence used for inference
        top_n: Number of top contributing factors to show
    
    Returns:
        Dictionary with explanation components
    """
    try:
        # Get base probability
        base_prob = infer_recommendation_prob(model, evidence)
        
        # Get marginal probabilities for each variable
        infer = VariableElimination(model)
        
        explanations = []
        
        # Test each variable's contribution
        for var in ['Sentiment', 'UserPreference', 'Genre_small', 'Popularity', 'ReadLength']:
            if var in evidence:
                # Get probability without this variable
                evidence_without = {k: v for k, v in evidence.items() if k != var}
                if evidence_without:
                    prob_without = infer_recommendation_prob(model, evidence_without)
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
        for i, exp in enumerate(explanations[:top_n]):
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
        print(f"Error generating explanation: {e}")
        return {
            'probability': 0.5,
            'top_contributors': [],
            'explanation': "Unable to generate explanation"
        }

def main():
    """Main function to test BN inference"""
    print("Bayesian Network Inference Testing")
    print("="*40)
    
    # Load model
    model = load_fitted_model()
    if model is None:
        return
    
    # Test inference with examples
    test_results = test_inference_examples(model)
    
    # Load candidates for re-ranking
    try:
        candidates_df = pd.read_csv('candidates_df.csv')
        print(f"\nLoaded {len(candidates_df)} candidates for re-ranking")
        
        # Example user evidence
        user_evidence = {
            'Sentiment': 'Positive',
            'UserPreference': 'GenreFan'
        }
        
        # Re-rank candidates
        reranked = rerank_candidates(model, candidates_df, user_evidence, alpha=0.7)
        
        print(f"\nTop 10 Re-ranked Recommendations:")
        print("-" * 50)
        for idx, row in reranked.head(10).iterrows():
            print(f"{row['rank']}. {row['title'][:50]}...")
            print(f"   DL Score: {row['dl_score']:.3f}, BN Prob: {row['bn_probability']:.3f}, Final: {row['final_score']:.3f}")
            print()
        
        # Save results
        reranked.to_csv('inference_results.csv', index=False)
        print("Re-ranking results saved as 'inference_results.csv'")
        
        # Generate explanations for top recommendations
        print("\nGenerating Explanations for Top Recommendations:")
        print("-" * 60)
        
        for idx, row in reranked.head(3).iterrows():
            evidence = user_evidence.copy()
            evidence.update({
                'Genre_small': row['genre'],
                'Popularity': row['popularity'],
                'ReadLength': 'medium'
            })
            
            explanation = explain_recommendation(model, evidence)
            
            print(f"\n{row['rank']}. {row['title'][:50]}...")
            print(f"   BN Probability: {explanation['probability']:.3f}")
            print(f"   Explanation: {explanation['explanation']}")
            print()
        
    except FileNotFoundError:
        print("candidates_df.csv not found. Skipping re-ranking test.")
    
    print("\nBN inference testing completed!")

if __name__ == "__main__":
    main()
