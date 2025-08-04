#!/usr/bin/env python3
"""
Championship V2 Solution - Beat the Leader!
Current best: 12,337,503.37
Our score: 12,652,242.57 (need to improve by ~315k)

Advanced techniques:
1. Optimized hyperparameters
2. Weighted ensemble optimization
3. Advanced feature interactions
4. Outlier-resistant preprocessing
5. Strategic cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df):
    """Create advanced feature interactions and transformations"""
    
    # Log transformations for skewed features
    skewed_features = ['Twitter Followers', 'Instagram Followers', 'Facebook Popularity']
    for feat in skewed_features:
        if feat in df.columns:
            df[f'{feat}_log'] = np.log1p(df[feat].fillna(0))
            df[f'{feat}_sqrt'] = np.sqrt(df[feat].fillna(0))
    
    # Advanced location interactions
    if 'City' in df.columns and 'State' in df.columns:
        df['City_State_Combo'] = df['City'].astype(str) + '_' + df['State'].astype(str)
        city_counts = df['City'].value_counts()
        df['City_Frequency'] = df['City'].map(city_counts)
        
    # Cuisine popularity and interactions
    if 'Cuisine' in df.columns:
        cuisine_counts = df['Cuisine'].value_counts()
        df['Cuisine_Popularity'] = df['Cuisine'].map(cuisine_counts)
        
        # Cuisine-Location interactions
        if 'State' in df.columns:
            df['Cuisine_State_Combo'] = df['Cuisine'].astype(str) + '_' + df['State'].astype(str)
    
    # Social media ratios and interactions
    social_cols = ['Twitter Followers', 'Instagram Followers', 'Facebook Popularity']
    available_social = [col for col in social_cols if col in df.columns]
    
    if len(available_social) >= 2:
        df['Social_Media_Total'] = df[available_social].fillna(0).sum(axis=1)
        df['Social_Media_Max'] = df[available_social].fillna(0).max(axis=1)
        df['Social_Media_Std'] = df[available_social].fillna(0).std(axis=1)
        
        for i, col1 in enumerate(available_social):
            for col2 in available_social[i+1:]:
                df[f'{col1}_{col2}_Ratio'] = (df[col1].fillna(0) + 1) / (df[col2].fillna(0) + 1)
    
    # Rating interactions
    rating_cols = ['Dining Rating', 'Delivery Rating']
    available_ratings = [col for col in rating_cols if col in df.columns]
    
    if len(available_ratings) >= 2:
        df['Rating_Avg'] = df[available_ratings].fillna(df[available_ratings].mean()).mean(axis=1)
        df['Rating_Diff'] = abs(df[available_ratings[0]].fillna(3.0) - df[available_ratings[1]].fillna(3.0))
        df['Rating_Product'] = df[available_ratings].fillna(3.0).prod(axis=1)
    
    # Advanced categorical interactions
    cat_cols = ['Cuisine', 'City', 'State']
    available_cats = [col for col in cat_cols if col in df.columns]
    
    # Create interaction features for top categories
    for col in available_cats:
        if col in df.columns:
            top_categories = df[col].value_counts().head(10).index.tolist()
            for cat in top_categories:
                df[f'{col}_is_{cat}'] = (df[col] == cat).astype(int)
    
    # Tier-based features
    if 'Tier' in df.columns:
        df['Tier_filled'] = df['Tier'].fillna('Unknown')
        tier_counts = df['Tier_filled'].value_counts()
        df['Tier_Frequency'] = df['Tier_filled'].map(tier_counts)
    
    # Price range interactions
    if 'Price range' in df.columns:
        price_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['Price_Numeric'] = df['Price range'].map(price_map).fillna(2)
        
        # Price-Cuisine interactions
        if 'Cuisine' in df.columns:
            df['Price_Cuisine_Interaction'] = df['Price_Numeric'] * pd.Categorical(df['Cuisine']).codes
    
    return df

def optimize_preprocessing(X_train, X_test, y_train):
    """Optimized preprocessing pipeline"""
    
    print("Starting advanced preprocessing...")
    
    # Combine for consistent preprocessing
    combined_df = pd.concat([X_train, X_test], ignore_index=True)
    
    # Advanced feature engineering
    combined_df = advanced_feature_engineering(combined_df)
    
    # Handle categorical variables with target encoding for high cardinality
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    
    label_encoders = {}
    for col in categorical_cols:
        if combined_df[col].nunique() > 50:  # High cardinality - use target encoding
            # Simple target encoding with regularization
            if col in X_train.columns:
                target_mean = y_train.mean()
                col_means = X_train.groupby(col)[y_train.name if hasattr(y_train, 'name') else 'target'].mean()
                col_counts = X_train[col].value_counts()
                
                # Regularized target encoding
                alpha = 10
                regularized_means = (col_counts * col_means + alpha * target_mean) / (col_counts + alpha)
                
                combined_df[f'{col}_target_encoded'] = combined_df[col].map(regularized_means).fillna(target_mean)
                combined_df.drop(col, axis=1, inplace=True)
        else:
            # Low cardinality - use label encoding
            le = LabelEncoder()
            combined_df[col] = le.fit_transform(combined_df[col].astype(str))
            label_encoders[col] = le
    
    # Split back
    X_train_processed = combined_df.iloc[:len(X_train)].copy()
    X_test_processed = combined_df.iloc[len(X_train):].copy()
    
    # Feature selection - keep top features
    selector = SelectKBest(score_func=f_regression, k=min(150, X_train_processed.shape[1]))
    X_train_selected = selector.fit_transform(X_train_processed.fillna(0), y_train)
    X_test_selected = selector.transform(X_test_processed.fillna(0))
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f"Final feature count: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, scaler, selector

def get_optimized_models():
    """Get models with optimized hyperparameters"""
    
    models = {
        'rf1': RandomForestRegressor(
            n_estimators=200, max_depth=25, min_samples_split=5,
            min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1
        ),
        'rf2': RandomForestRegressor(
            n_estimators=150, max_depth=30, min_samples_split=3,
            min_samples_leaf=1, max_features=0.7, random_state=123, n_jobs=-1
        ),
        'et1': ExtraTreesRegressor(
            n_estimators=200, max_depth=25, min_samples_split=4,
            min_samples_leaf=1, max_features=0.9, random_state=42, n_jobs=-1
        ),
        'et2': ExtraTreesRegressor(
            n_estimators=180, max_depth=35, min_samples_split=2,
            min_samples_leaf=2, max_features=0.8, random_state=456, n_jobs=-1
        ),
        'gbm1': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=8,
            min_samples_split=5, min_samples_leaf=3, subsample=0.8,
            max_features=0.7, random_state=42
        ),
        'gbm2': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=10,
            min_samples_split=4, min_samples_leaf=2, subsample=0.85,
            max_features=0.8, random_state=789
        ),
        'ridge1': Ridge(alpha=50.0),
        'ridge2': Ridge(alpha=100.0),
        'elastic1': ElasticNet(alpha=10.0, l1_ratio=0.7, max_iter=2000),
        'elastic2': ElasticNet(alpha=20.0, l1_ratio=0.5, max_iter=2000),
        'huber1': HuberRegressor(epsilon=1.5, alpha=0.001, max_iter=200),
        'huber2': HuberRegressor(epsilon=2.0, alpha=0.01, max_iter=200),
    }
    
    return models

def optimize_ensemble_weights(models, X_val, y_val):
    """Optimize ensemble weights using validation set"""
    
    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_val)
    
    # Try different weight combinations
    best_rmse = float('inf')
    best_weights = None
    
    # Grid search for weights
    from itertools import product
    
    # Focus on top performing models
    model_names = list(models.keys())
    
    # Simple optimization: try different weight schemes
    weight_schemes = [
        # Equal weights
        {name: 1.0/len(models) for name in model_names},
        
        # Tree-heavy
        {name: 0.15 if 'rf' in name or 'et' in name else 0.08 if 'gbm' in name else 0.05 
         for name in model_names},
        
        # GBM-heavy
        {name: 0.2 if 'gbm' in name else 0.12 if 'rf' in name or 'et' in name else 0.05
         for name in model_names},
        
        # Balanced
        {name: 0.12 if any(x in name for x in ['rf', 'et', 'gbm']) else 0.08
         for name in model_names}
    ]
    
    for weights in weight_schemes:
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate ensemble prediction
        ensemble_pred = np.zeros(len(y_val))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]
        
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights.copy()
    
    print(f"Best validation RMSE: {best_rmse:,.2f}")
    print("Optimal weights:", {k: f"{v:.3f}" for k, v in best_weights.items()})
    
    return best_weights

def main():
    """Main execution function"""
    
    print("=== Championship V2 Solution - Beat the Leader! ===")
    start_time = datetime.now()
    
    # Load data
    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Fix column name inconsistency
    if 'Endoresed By' in test_df.columns and 'Endorsed By' in train_df.columns:
        test_df.rename(columns={'Endoresed By': 'Endorsed By'}, inplace=True)
    
    # Prepare features and target
    target_col = 'Annual Turnover'
    feature_cols = [col for col in train_df.columns if col not in [target_col, 'Registration Number']]
    
    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    
    # Remove outliers from training (keep extreme values in test)
    Q1 = y.quantile(0.01)
    Q3 = y.quantile(0.99)
    outlier_mask = (y >= Q1) & (y <= Q3)
    
    X_clean = X[outlier_mask].copy()
    y_clean = y[outlier_mask].copy()
    
    print(f"Removed {len(X) - len(X_clean)} outliers")
    
    # Split with stratification based on target quantiles
    y_bins = pd.qcut(y_clean, q=5, labels=False, duplicates='drop')
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.15, random_state=42, stratify=y_bins
    )
    
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
    
    # Advanced preprocessing
    X_train_processed, X_test_processed, scaler, selector = optimize_preprocessing(X_train, X_test, y_train)
    X_val_processed = selector.transform(X_val.fillna(0))
    X_val_processed = scaler.transform(X_val_processed)
    
    # Train optimized models
    print("\nTraining optimized models...")
    models = get_optimized_models()
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        
        # Validation score
        val_pred = model.predict(X_val_processed)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  {name} validation RMSE: {val_rmse:,.2f}")
    
    # Optimize ensemble weights
    print("\nOptimizing ensemble weights...")
    best_weights = optimize_ensemble_weights(trained_models, X_val_processed, y_val)
    
    # Final predictions
    print("\nGenerating final predictions...")
    final_predictions = np.zeros(len(X_test_processed))
    
    for name, model in trained_models.items():
        pred = model.predict(X_test_processed)
        final_predictions += best_weights[name] * pred
    
    # Create submission
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': final_predictions
    })
    
    submission_file = 'championship_v2_submission.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n=== Results ===")
    print(f"Submission saved to: {submission_file}")
    print(f"Predictions range: {final_predictions.min():,.0f} to {final_predictions.max():,.0f}")
    print(f"Mean prediction: {final_predictions.mean():,.0f}")
    print(f"Processing time: {datetime.now() - start_time}")
    
    print(f"\n🎯 Target: Beat 12,337,503.37 RMSE")
    print(f"📊 Previous score: 12,652,242.57")
    print(f"🚀 Improvements made:")
    print(f"   • Optimized hyperparameters")
    print(f"   • Advanced feature engineering")
    print(f"   • Weighted ensemble optimization")
    print(f"   • Outlier-resistant preprocessing")
    print(f"   • Strategic feature selection")

if __name__ == "__main__":
    main()