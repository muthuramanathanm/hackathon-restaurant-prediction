#!/usr/bin/env python3
"""
Beat the Leader Solution!
Target: Beat 12,337,503.37 RMSE
Current: 12,652,242.57 RMSE (need to improve by ~315k)

Key optimizations:
1. Hyperparameter-tuned models
2. Smart ensemble weighting
3. Advanced feature engineering
4. Robust preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Advanced feature engineering"""
    
    # Log transforms for skewed social media features
    social_cols = ['Twitter Followers', 'Instagram Followers', 'Facebook Popularity']
    for col in social_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))
            df[f'{col}_sqrt'] = np.sqrt(df[col].fillna(0))
    
    # Social media combinations
    if all(col in df.columns for col in social_cols):
        df['Social_Total'] = df[social_cols].fillna(0).sum(axis=1)
        df['Social_Max'] = df[social_cols].fillna(0).max(axis=1)
        df['Social_Mean'] = df[social_cols].fillna(0).mean(axis=1)
        
        # Ratios
        df['Twitter_Instagram_Ratio'] = (df['Twitter Followers'].fillna(0) + 1) / (df['Instagram Followers'].fillna(0) + 1)
        df['Facebook_Social_Ratio'] = (df['Facebook Popularity'].fillna(0) + 1) / (df['Social_Total'] + 1)
    
    # Rating features
    rating_cols = ['Dining Rating', 'Delivery Rating']
    if all(col in df.columns for col in rating_cols):
        df['Rating_Mean'] = df[rating_cols].fillna(3.0).mean(axis=1)
        df['Rating_Diff'] = abs(df['Dining Rating'].fillna(3.0) - df['Delivery Rating'].fillna(3.0))
        df['Rating_Product'] = df[rating_cols].fillna(3.0).prod(axis=1)
        df['Rating_Max'] = df[rating_cols].fillna(3.0).max(axis=1)
    
    # Location frequency encoding
    if 'City' in df.columns:
        city_counts = df['City'].value_counts()
        df['City_Frequency'] = df['City'].map(city_counts).fillna(1)
        
        # Rare cities indicator
        df['City_Rare'] = (df['City_Frequency'] <= 5).astype(int)
    
    if 'State' in df.columns:
        state_counts = df['State'].value_counts()
        df['State_Frequency'] = df['State'].map(state_counts).fillna(1)
    
    # Cuisine frequency
    if 'Cuisine' in df.columns:
        cuisine_counts = df['Cuisine'].value_counts()
        df['Cuisine_Frequency'] = df['Cuisine'].map(cuisine_counts).fillna(1)
        df['Cuisine_Rare'] = (df['Cuisine_Frequency'] <= 10).astype(int)
    
    # Price range numerical
    if 'Price range' in df.columns:
        price_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['Price_Numeric'] = df['Price range'].map(price_map).fillna(2)
    
    # Tier handling
    if 'Tier' in df.columns:
        df['Tier_Missing'] = df['Tier'].isnull().astype(int)
        df['Tier'].fillna('Unknown', inplace=True)
    
    # Combined features
    if 'Cuisine' in df.columns and 'State' in df.columns:
        df['Cuisine_State'] = df['Cuisine'].astype(str) + '_' + df['State'].astype(str)
    
    if 'City' in df.columns and 'State' in df.columns:
        df['City_State'] = df['City'].astype(str) + '_' + df['State'].astype(str)
    
    # Interaction with price
    if 'Price_Numeric' in df.columns and 'Social_Total' in df.columns:
        df['Price_Social_Interaction'] = df['Price_Numeric'] * np.log1p(df['Social_Total'])
    
    if 'Price_Numeric' in df.columns and 'Rating_Mean' in df.columns:
        df['Price_Rating_Interaction'] = df['Price_Numeric'] * df['Rating_Mean']
    
    return df

def preprocess_data(X_train, X_test, y_train):
    """Optimized preprocessing pipeline"""
    
    print("Advanced feature engineering...")
    
    # Combine for consistent preprocessing
    train_size = len(X_train)
    combined = pd.concat([X_train, X_test], ignore_index=True)
    
    # Feature engineering
    combined = engineer_features(combined)
    
    # Handle categorical variables
    categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
    
    # Simple target encoding for high-cardinality categories
    for col in categorical_cols:
        if combined[col].nunique() > 20:
            # Create target encoding using training data only
            train_data = X_train.copy()
            train_data['target'] = y_train
            
            col_stats = train_data.groupby(col)['target'].agg(['mean', 'count']).reset_index()
            global_mean = y_train.mean()
            
            # Smoothing
            alpha = 100
            col_stats['target_encoded'] = (col_stats['count'] * col_stats['mean'] + alpha * global_mean) / (col_stats['count'] + alpha)
            
            # Map to combined data
            encoding_dict = dict(zip(col_stats[col], col_stats['target_encoded']))
            combined[f'{col}_encoded'] = combined[col].map(encoding_dict).fillna(global_mean)
            combined.drop(col, axis=1, inplace=True)
        else:
            # Label encoding for low cardinality
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Split back
    X_train_processed = combined.iloc[:train_size].copy()
    X_test_processed = combined.iloc[train_size:].copy()
    
    # Fill missing values
    X_train_processed = X_train_processed.fillna(X_train_processed.median())
    X_test_processed = X_test_processed.fillna(X_train_processed.median())
    
    print(f"Features after engineering: {X_train_processed.shape[1]}")
    
    return X_train_processed, X_test_processed

def get_champion_models():
    """Optimized models with best hyperparameters"""
    
    return {
        'rf_best': RandomForestRegressor(
            n_estimators=250, max_depth=30, min_samples_split=3,
            min_samples_leaf=1, max_features=0.8, random_state=42, n_jobs=-1
        ),
        'rf_deep': RandomForestRegressor(
            n_estimators=200, max_depth=35, min_samples_split=2,
            min_samples_leaf=2, max_features=0.7, random_state=123, n_jobs=-1
        ),
        'et_best': ExtraTreesRegressor(
            n_estimators=250, max_depth=28, min_samples_split=3,
            min_samples_leaf=1, max_features=0.9, random_state=42, n_jobs=-1
        ),
        'et_diverse': ExtraTreesRegressor(
            n_estimators=200, max_depth=40, min_samples_split=2,
            min_samples_leaf=3, max_features=0.8, random_state=456, n_jobs=-1
        ),
        'gbm_precise': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.06, max_depth=9,
            min_samples_split=4, min_samples_leaf=2, subsample=0.85,
            max_features=0.8, random_state=42
        ),
        'gbm_robust': GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=12,
            min_samples_split=3, min_samples_leaf=1, subsample=0.9,
            max_features=0.7, random_state=789
        ),
        'ridge_strong': Ridge(alpha=75.0),
        'ridge_smooth': Ridge(alpha=150.0),
        'elastic_balanced': ElasticNet(alpha=15.0, l1_ratio=0.6, max_iter=2000),
        'elastic_sparse': ElasticNet(alpha=25.0, l1_ratio=0.8, max_iter=2000),
        'huber_robust': HuberRegressor(epsilon=1.8, alpha=0.005, max_iter=300),
        'lasso_select': Lasso(alpha=500.0, max_iter=2000)
    }

def optimize_weights(models, X_val, y_val):
    """Find optimal ensemble weights"""
    
    # Get all predictions
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X_val)
    
    # Test different weighting schemes
    schemes = [
        # Tree models heavy
        {'rf_best': 0.18, 'rf_deep': 0.16, 'et_best': 0.16, 'et_diverse': 0.14,
         'gbm_precise': 0.14, 'gbm_robust': 0.12, 'ridge_strong': 0.04, 'ridge_smooth': 0.02,
         'elastic_balanced': 0.02, 'elastic_sparse': 0.01, 'huber_robust': 0.01, 'lasso_select': 0.0},
        
        # GBM heavy
        {'rf_best': 0.12, 'rf_deep': 0.10, 'et_best': 0.12, 'et_diverse': 0.10,
         'gbm_precise': 0.22, 'gbm_robust': 0.20, 'ridge_strong': 0.06, 'ridge_smooth': 0.04,
         'elastic_balanced': 0.02, 'elastic_sparse': 0.01, 'huber_robust': 0.01, 'lasso_select': 0.0},
        
        # Balanced ensemble
        {'rf_best': 0.15, 'rf_deep': 0.13, 'et_best': 0.15, 'et_diverse': 0.12,
         'gbm_precise': 0.16, 'gbm_robust': 0.14, 'ridge_strong': 0.06, 'ridge_smooth': 0.04,
         'elastic_balanced': 0.03, 'elastic_sparse': 0.01, 'huber_robust': 0.01, 'lasso_select': 0.0},
        
        # Conservative blend
        {'rf_best': 0.20, 'rf_deep': 0.16, 'et_best': 0.18, 'et_diverse': 0.14,
         'gbm_precise': 0.12, 'gbm_robust': 0.10, 'ridge_strong': 0.05, 'ridge_smooth': 0.03,
         'elastic_balanced': 0.01, 'elastic_sparse': 0.01, 'huber_robust': 0.0, 'lasso_select': 0.0}
    ]
    
    best_rmse = float('inf')
    best_weights = None
    
    for i, weights in enumerate(schemes):
        ensemble_pred = np.zeros(len(y_val))
        for name, weight in weights.items():
            if name in preds:
                ensemble_pred += weight * preds[name]
        
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        print(f"Scheme {i+1} RMSE: {rmse:,.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights
    
    print(f"Best validation RMSE: {best_rmse:,.2f}")
    return best_weights

def main():
    """Execute championship solution"""
    
    print("🏆 BEAT THE LEADER SOLUTION 🏆")
    print(f"Target: Beat 12,337,503.37")
    print(f"Current: 12,652,242.57")
    print(f"Gap to close: ~315,000")
    
    start_time = datetime.now()
    
    # Load data
    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')
    
    # Fix column name issue
    if 'Endoresed By' in test_df.columns and 'Endorsed By' in train_df.columns:
        test_df.rename(columns={'Endoresed By': 'Endorsed By'}, inplace=True)
    
    # Prepare data
    target_col = 'Annual Turnover'
    feature_cols = [col for col in train_df.columns if col not in [target_col, 'Registration Number']]
    
    X = train_df[feature_cols].copy()
    y = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    
    print(f"Original data - Train: {X.shape}, Test: {X_test.shape}")
    
    # Remove extreme outliers (keep more data than before)
    Q1, Q3 = y.quantile([0.005, 0.995])  # Keep 99% of data
    mask = (y >= Q1) & (y <= Q3)
    X_clean, y_clean = X[mask].copy(), y[mask].copy()
    
    print(f"After outlier removal: {len(X_clean)} samples ({len(X) - len(X_clean)} removed)")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # Preprocessing
    X_train_proc, X_test_proc = preprocess_data(X_train, X_test, y_train)
    X_val_proc, _ = preprocess_data(X_val, X_test, y_val)  # Use same preprocessing
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)
    X_test_scaled = scaler.transform(X_test_proc)
    
    # Train models
    print("\nTraining champion models...")
    models = get_champion_models()
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # Validation score
        val_pred = model.predict(X_val_scaled)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  {name}: {rmse:,.2f}")
    
    # Optimize ensemble
    print("\nOptimizing ensemble weights...")
    best_weights = optimize_weights(trained_models, X_val_scaled, y_val)
    
    # Final predictions
    print("\nGenerating final predictions...")
    final_preds = np.zeros(X_test_scaled.shape[0])
    
    for name, weight in best_weights.items():
        if name in trained_models and weight > 0:
            pred = trained_models[name].predict(X_test_scaled)
            final_preds += weight * pred
            print(f"{name}: {weight:.3f}")
    
    # Create submission
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': final_preds
    })
    
    filename = 'beat_leader_submission.csv'
    submission.to_csv(filename, index=False)
    
    # Results
    print(f"\n🎯 RESULTS:")
    print(f"Submission: {filename}")
    print(f"Range: {final_preds.min():,.0f} to {final_preds.max():,.0f}")
    print(f"Mean: {final_preds.mean():,.0f}")
    print(f"Time: {datetime.now() - start_time}")
    
    print(f"\n🚀 Optimizations applied:")
    print(f"  ✓ Kept 99% of training data (less aggressive outlier removal)")
    print(f"  ✓ Advanced feature engineering ({X_train_proc.shape[1]} features)")
    print(f"  ✓ Optimized hyperparameters for all 12 models")
    print(f"  ✓ Smart ensemble weighting")
    print(f"  ✓ Robust scaling and preprocessing")
    
    print(f"\n🏆 GO BEAT THAT LEADER! 🏆")

if __name__ == "__main__":
    main()