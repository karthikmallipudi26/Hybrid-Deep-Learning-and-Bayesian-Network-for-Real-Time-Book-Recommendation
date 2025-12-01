# Hybrid DL-BN Book Recommendation System

A hybrid book recommendation system that combines Deep Learning (DL) for candidate generation with Bayesian Networks (BN) for probabilistic re-ranking and personalization.

## Project Structure

```
├── Data Files
│   ├── books_cleaned_bn.csv          # Processed book data (5,199 books)
│   ├── candidates_df.csv             # DL-generated candidates
│   ├── user_preferences.csv          # User preference clusters (50 users)
│   ├── bn_data_balanced.csv          # Balanced training data (1,540 examples)
│   └── fitted_bn.pkl                 # Trained Bayesian Network model
│
├── Core Implementation
│   ├── bn_build_and_fit.py           # BN model creation and training
│   ├── bn_inference.py               # BN inference and testing
│   ├── dl_bn_integration.py          # Hybrid system integration
│   └── evaluation.py                 # Performance evaluation
│
├── Demonstration
│   ├── final_demo.py                 # Complete system demonstration
│   ├── show_cpts.py                  # Display CPTs
│   └── test_bn.py                    # BN testing
│
└── Documentation
    ├── REVIEW_SUMMARY.md             # Comprehensive review summary
    ├── workflow.docx                 # Project workflow
    └── progress.docx                 # Progress tracking
```

## System Architecture

### 1. Deep Learning Component
- **Embedding Model**: all-MiniLM-L6-v2 transformer
- **Purpose**: Generate semantic similarity scores
- **Output**: Top-N candidates with DL scores

### 2. Bayesian Network Component
- **Structure**: 6 nodes, 9 edges
- **Nodes**: Sentiment, UserPreference, Genre_small, Popularity, ReadLength, Recommendation
- **Purpose**: Compute P(Recommendation=Yes | evidence)

### 3. Hybrid Integration
- **Formula**: Final Score = DL_Score × (α × BN_Probability + (1-α))
- **α Parameter**: Controls BN influence (typically 0.7)

## Quick Start

### 1. Run the Complete Demonstration
```bash
python final_demo.py
```

### 2. View Conditional Probability Tables
```bash
python show_cpts.py
```

### 3. Test BN Inference
```bash
python test_bn.py
```

## Key Features

✅ **Bayesian Network Implementation**
- 6-node network with realistic structure
- CPTs generated from training data
- Inference working with realistic probabilities

✅ **Data Processing**
- Sentiment analysis integration
- User preference clustering (4 types)
- Book feature engineering

✅ **Hybrid Integration**
- DL-BN score combination
- Re-ranking demonstration
- Personalization through user preferences

## Bayesian Network Structure

```
Nodes: ['Sentiment', 'Recommendation', 'UserPreference', 'Genre_small', 'Popularity', 'ReadLength']

Edges:
- Sentiment → Recommendation
- Sentiment → Genre_small
- UserPreference → Recommendation
- UserPreference → Genre_small
- UserPreference → Popularity
- UserPreference → ReadLength
- Genre_small → Recommendation
- Popularity → Recommendation
- ReadLength → Recommendation
```

## User Preference Clusters

1. **GenreFan** (44%): Prefers specific genres, high popularity books
2. **AuthorLoyal** (16%): Follows favorite authors, longer books
3. **CasualReader** (18%): Short books, diverse reading
4. **DiverseReader** (22%): Wide variety of genres and lengths

## Inference Examples

### High Recommendation Case
- **Evidence**: Positive sentiment, GenreFan, Fiction, High popularity
- **P(Recommendation=Yes)**: ≈ 0.85

### Low Recommendation Case
- **Evidence**: Negative sentiment, CasualReader, Non-fiction, Low popularity
- **P(Recommendation=Yes)**: ≈ 0.25

## Re-ranking Process

1. **DL Generation**: Get Top-N candidates with semantic similarity scores
2. **BN Inference**: Compute P(Recommendation=Yes | evidence) for each candidate
3. **Score Combination**: Final Score = DL_Score × (α × BN_Probability + (1-α))
4. **Re-ranking**: Sort by final score and return Top-K recommendations

## Files for Review

### Essential Files
- `fitted_bn.pkl`: Trained Bayesian Network model
- `bn_data_balanced.csv`: Training data with synthetic examples
- `final_demo.py`: Complete system demonstration
- `REVIEW_SUMMARY.md`: Comprehensive review documentation

### Data Files
- `candidates_df.csv`: DL-generated candidates (50 books)
- `user_preferences.csv`: User preference clusters (50 users)
- `books_cleaned_bn.csv`: Processed book data (5,199 books)

## Technical Details

### BN Training
- **Estimator**: BayesianEstimator with BDeu prior
- **Equivalent Sample Size**: 100 for smooth CPTs
- **Training Data**: 1,540 examples (540 real + 1,000 synthetic)

### Inference
- **Method**: Variable Elimination
- **Evidence**: User preferences + book features
- **Output**: P(Recommendation=Yes | evidence)

### Integration
- **α Parameter**: 0.7 (70% BN influence, 30% DL influence)
- **Re-ranking**: Combines semantic similarity with personalization
- **Explanation**: Generates reasoning for recommendations

## Evaluation Metrics

- **Precision@K**: Fraction of relevant recommendations in top-K
- **Recall@K**: Fraction of relevant items retrieved
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain

## Next Steps

1. **Complete Integration**: Connect all components in a single pipeline
2. **Evaluation**: Run comprehensive evaluation with test data
3. **Optimization**: Tune α parameter and BN structure
4. **UI Development**: Create user interface for recommendations
5. **Performance**: Optimize for real-time inference

## Review Focus

For the first review, the key deliverables are:

1. **BN Model**: Working Bayesian Network with CPTs
2. **Inference**: Realistic probability calculations
3. **Integration**: DL-BN score combination
4. **Data**: Processed datasets ready for training
5. **Demonstration**: Re-ranking examples

The system demonstrates the core concepts of hybrid recommendation systems combining semantic similarity (DL) with probabilistic reasoning (BN) for personalized book recommendations.
