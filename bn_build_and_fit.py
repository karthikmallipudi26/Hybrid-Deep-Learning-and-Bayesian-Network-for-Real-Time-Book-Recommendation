import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_bn_structure():
    """Create the Bayesian Network structure"""
    # Define the BN structure with 6 nodes and 9 edges
    edges = [
        ('Sentiment', 'Recommendation'),
        ('UserPreference', 'Recommendation'),
        ('Genre_small', 'Recommendation'),
        ('Popularity', 'Recommendation'),
        ('ReadLength', 'Recommendation'),
        ('UserPreference', 'Genre_small'),
        ('UserPreference', 'Popularity'),
        ('UserPreference', 'ReadLength'),
        ('Sentiment', 'Genre_small')
    ]
    
    model = DiscreteBayesianNetwork(edges)
    return model

def load_and_prepare_data():
    """Load and prepare the data for BN training"""
    print("Loading data...")
    
    # Load the main data
    books_df = pd.read_csv('books_cleaned_bn.csv')
    user_prefs = pd.read_csv('user_preferences.csv')
    interactions = pd.read_csv('user_item_interaction.py') if 'user_item_interaction.py' in globals() else None
    
    # Load existing BN data if available
    try:
        bn_data = pd.read_csv('bn_data.csv')
        print(f"Loaded existing BN data: {len(bn_data)} rows")
    except:
        bn_data = None
    
    return books_df, user_prefs, bn_data

def create_synthetic_data(books_df, user_prefs, target_size=1000):
    """Create balanced synthetic data for BN training"""
    print("Creating synthetic data...")
    
    # Define valid states for each variable
    sentiment_states = ['Positive', 'Neutral', 'Negative']
    user_pref_states = ['GenreFan', 'AuthorLoyal', 'CasualReader', 'DiverseReader']
    genre_states = books_df['genre'].unique()[:10]  # Top 10 genres
    popularity_states = ['Low', 'Medium', 'High', 'Very High']
    read_length_states = ['short', 'medium', 'long', 'unknown']
    recommendation_states = ['Yes', 'No']
    
    synthetic_data = []
    
    # Create synthetic data with realistic conditional probabilities
    for i in range(target_size):
        # Sample user preference
        user_pref = np.random.choice(user_pref_states, p=[0.44, 0.16, 0.18, 0.22])
        
        # Sample sentiment based on user preference
        if user_pref == 'GenreFan':
            sentiment = np.random.choice(sentiment_states, p=[0.4, 0.4, 0.2])
        elif user_pref == 'AuthorLoyal':
            sentiment = np.random.choice(sentiment_states, p=[0.5, 0.3, 0.2])
        elif user_pref == 'CasualReader':
            sentiment = np.random.choice(sentiment_states, p=[0.3, 0.5, 0.2])
        else:  # DiverseReader
            sentiment = np.random.choice(sentiment_states, p=[0.35, 0.45, 0.2])
        
        # Sample genre based on user preference
        if user_pref == 'GenreFan':
            genre = np.random.choice(genre_states, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01])
        elif user_pref == 'AuthorLoyal':
            genre = np.random.choice(genre_states, p=[0.25, 0.25, 0.2, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01, 0.0])
        else:
            genre = np.random.choice(genre_states, p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
        
        # Sample popularity based on user preference
        if user_pref in ['GenreFan', 'AuthorLoyal']:
            popularity = np.random.choice(popularity_states, p=[0.1, 0.2, 0.3, 0.4])
        else:
            popularity = np.random.choice(popularity_states, p=[0.2, 0.3, 0.3, 0.2])
        
        # Sample read length based on user preference
        if user_pref == 'CasualReader':
            read_length = np.random.choice(read_length_states, p=[0.4, 0.3, 0.2, 0.1])
        elif user_pref == 'AuthorLoyal':
            read_length = np.random.choice(read_length_states, p=[0.2, 0.3, 0.4, 0.1])
        else:
            read_length = np.random.choice(read_length_states, p=[0.25, 0.35, 0.3, 0.1])
        
        # Sample recommendation based on all evidence
        # Higher probability for positive sentiment and matching preferences
        base_prob = 0.3
        if sentiment == 'Positive':
            base_prob += 0.3
        elif sentiment == 'Neutral':
            base_prob += 0.1
        
        if user_pref == 'GenreFan' and popularity in ['High', 'Very High']:
            base_prob += 0.2
        elif user_pref == 'AuthorLoyal' and read_length in ['medium', 'long']:
            base_prob += 0.2
        elif user_pref == 'CasualReader' and read_length == 'short':
            base_prob += 0.2
        elif user_pref == 'DiverseReader':
            base_prob += 0.1
        
        # Ensure probability is between 0 and 1
        base_prob = min(0.9, max(0.1, base_prob))
        
        recommendation = 'Yes' if np.random.random() < base_prob else 'No'
        
        synthetic_data.append({
            'Sentiment': sentiment,
            'UserPreference': user_pref,
            'Genre_small': genre,
            'Popularity': popularity,
            'ReadLength': read_length,
            'Recommendation': recommendation,
            '__synthetic__': True
        })
    
    return pd.DataFrame(synthetic_data)

def fit_bn_model(model, data):
    """Fit the Bayesian Network model"""
    print("Fitting BN model...")
    
    # Use BayesianEstimator with higher equivalent sample size for smoother CPTs
    estimator = BayesianEstimator(model, data)
    fitted_model = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=100)
    
    # Update model with fitted parameters
    model.add_cpds(*fitted_model)
    
    return model

def print_cpds(model):
    """Print all Conditional Probability Distributions"""
    print("\n" + "="*50)
    print("CONDITIONAL PROBABILITY TABLES (CPTs)")
    print("="*50)
    
    for cpd in model.get_cpds():
        print(f"\nCPD for {cpd.variable}:")
        print("-" * 30)
        print(cpd)
        print()

def save_model_and_data(model, data):
    """Save the fitted model and data"""
    print("Saving model and data...")
    
    # Save fitted model
    with open('fitted_bn.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save balanced data
    data.to_csv('bn_data_balanced.csv', index=False)
    
    print("Model saved as 'fitted_bn.pkl'")
    print("Balanced data saved as 'bn_data_balanced.csv'")

def main():
    """Main function to build and fit the BN"""
    print("Building and Fitting Bayesian Network")
    print("="*40)
    
    # Load data
    books_df, user_prefs, bn_data = load_and_prepare_data()
    
    # Create synthetic data
    synthetic_data = create_synthetic_data(books_df, user_prefs, target_size=1000)
    
    # Combine with existing data if available
    if bn_data is not None:
        bn_data['__synthetic__'] = False
        combined_data = pd.concat([bn_data, synthetic_data], ignore_index=True)
    else:
        combined_data = synthetic_data
    
    print(f"Total training data: {len(combined_data)} rows")
    print(f"Synthetic data: {len(synthetic_data)} rows")
    
    # Create BN structure
    model = create_bn_structure()
    
    # Fit the model
    fitted_model = fit_bn_model(model, combined_data)
    
    # Print CPTs
    print_cpds(fitted_model)
    
    # Save model and data
    save_model_and_data(fitted_model, combined_data)
    
    print("\nBN model building and fitting completed successfully!")
    
    return fitted_model, combined_data

if __name__ == "__main__":
    model, data = main()
