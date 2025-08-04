#!/usr/bin/env python3
"""
Championship Restaurant Turnover Prediction
Advanced ensemble with target encoding, stacking, and optimized features
Goal: Beat RMSE of 12,337,503.37
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TargetEncoder:
    """Advanced target encoding with regularization"""

    def __init__(self, smoothing=1.0, min_samples_leaf=1, noise_level=0.01):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.encodings = {}
        self.global_mean = None

    def encode(self, train_series, target, test_series):
        """Encode categorical variable with target mean"""

        # Calculate global mean
        self.global_mean = target.mean()

        # Calculate category means and counts
        cat_stats = train_series.groupby(train_series).agg({
            target.name: ['mean', 'count']
        }).reset_index()
        cat_stats.columns = [train_series.name, 'mean', 'count']

        # Apply smoothing
        smoothing = 1 / (1 + np.exp(-(cat_stats['count'] - self.min_samples_leaf) / self.smoothing))
        cat_stats['smoothed_mean'] = (
            cat_stats['mean'] * smoothing + self.global_mean * (1 - smoothing)
        )

        # Create encoding dictionary
        encoding_dict = cat_stats.set_index(train_series.name)['smoothed_mean'].to_dict()

        # Encode train set with noise
        train_encoded = train_series.map(encoding_dict).fillna(self.global_mean)
        if self.noise_level > 0:
            noise = np.random.normal(0, target.std() * self.noise_level, len(train_encoded))
            train_encoded += noise

        # Encode test set
        test_encoded = test_series.map(encoding_dict).fillna(self.global_mean)

        return train_encoded, test_encoded

def load_and_clean_data():
    """Load and perform initial cleaning"""

    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Combine for preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Annual Turnover'] = np.nan

    # Fix column name inconsistency
    if 'Endoresed By' in test_df.columns:
        test_df['Endorsed By'] = test_df['Endoresed By']
        test_df = test_df.drop('Endoresed By', axis=1)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    return combined_df, train_df.shape[0]

def champion_feature_engineering(df):
    """Championship-level feature engineering"""

    # Handle missing values intelligently
    df['Facebook Popularity Quotient'] = df['Facebook Popularity Quotient'].fillna(df['Facebook Popularity Quotient'].median())
    df['Instagram Popularity Quotient'] = df['Instagram Popularity Quotient'].fillna(df['Instagram Popularity Quotient'].median())

    # City cleaning - handle -1 and missing
    df['City'] = df['City'].replace('-1', 'Unknown')
    df['City'] = df['City'].fillna('Unknown')

    # Advanced date parsing
    df['Opening Day of Restaurant'] = pd.to_datetime(df['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
    df['opening_year'] = df['Opening Day of Restaurant'].dt.year
    df['opening_month'] = df['Opening Day of Restaurant'].dt.month
    df['opening_day'] = df['Opening Day of Restaurant'].dt.day
    df['opening_weekday'] = df['Opening Day of Restaurant'].dt.dayofweek
    df['restaurant_age'] = 2024 - df['opening_year']
    df['restaurant_age'] = np.where(df['restaurant_age'] < 0, 0, df['restaurant_age'])

    # Restaurant lifecycle features
    df['is_new_restaurant'] = (df['restaurant_age'] <= 2).astype(int)
    df['is_mature_restaurant'] = (df['restaurant_age'] >= 10).astype(int)
    df['age_squared'] = df['restaurant_age'] ** 2
    df['age_log'] = np.log1p(df['restaurant_age'])

    # Seasonal opening effects
    df['opened_in_summer'] = ((df['opening_month'] >= 4) & (df['opening_month'] <= 6)).astype(int)
    df['opened_in_winter'] = ((df['opening_month'] >= 10) | (df['opening_month'] <= 2)).astype(int)
    df['opened_weekend'] = (df['opening_weekday'] >= 5).astype(int)

    # Handle endorsement
    df['Endorsed By'] = df['Endorsed By'].fillna('Not Specific')

    # Advanced cuisine analysis
    df['Cuisine'] = df['Cuisine'].fillna('unknown')
    df['cuisine_count'] = df['Cuisine'].str.count(',') + 1

    # Cuisine categories
    popular_cuisines = ['indian', 'italian', 'chinese', 'greek', 'thai', 'japanese', 'american']
    for cuisine in popular_cuisines:
        df[f'has_{cuisine}'] = df['Cuisine'].str.contains(cuisine, case=False, na=False).astype(int)

    # Premium cuisine features
    premium_cuisines = ['japanese', 'french', 'italian', 'mediterranean']
    ethnic_cuisines = ['indian', 'chinese', 'thai', 'korean', 'vietnamese']
    western_cuisines = ['american', 'british', 'irish', 'german']

    df['has_premium_cuisine'] = df['Cuisine'].str.lower().str.contains('|'.join(premium_cuisines), na=False).astype(int)
    df['has_ethnic_cuisine'] = df['Cuisine'].str.lower().str.contains('|'.join(ethnic_cuisines), na=False).astype(int)
    df['has_western_cuisine'] = df['Cuisine'].str.lower().str.contains('|'.join(western_cuisines), na=False).astype(int)

    # Location analysis
    df['is_business_hub'] = (df['Restaurant Location'] == 'Near Business Hub').astype(int)
    df['is_party_hub'] = (df['Restaurant Location'] == 'Near Party Hub').astype(int)

    # Social media features
    df['social_media_avg'] = (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient']) / 2
    df['social_media_max'] = np.maximum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_media_min'] = np.minimum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_media_diff'] = df['Facebook Popularity Quotient'] - df['Instagram Popularity Quotient']
    df['social_media_ratio'] = df['Facebook Popularity Quotient'] / (df['Instagram Popularity Quotient'] + 1)

    # Social media clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    social_features = df[['Facebook Popularity Quotient', 'Instagram Popularity Quotient']].fillna(0)
    df['social_cluster'] = kmeans.fit_predict(social_features)

    # Rating features with advanced imputation
    rating_cols = ['Order Wait Time', 'Staff Responsivness', 'Value for Money',
                   'Hygiene Rating', 'Food Rating', 'Overall Restaurant Rating',
                   'Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating',
                   'Live Sports Rating', 'Ambience', 'Lively', 'Service',
                   'Comfortablility', 'Privacy']

    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Smart imputation based on restaurant type and tier
    for col in rating_cols:
        df[col] = df.groupby(['Restaurant Type', 'Resturant Tier'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    # Advanced composite scores
    service_cols = ['Staff Responsivness', 'Service']
    df['service_excellence'] = df[service_cols].mean(axis=1)

    quality_cols = ['Food Rating', 'Value for Money', 'Hygiene Rating']
    df['quality_score'] = df[quality_cols].mean(axis=1)

    experience_cols = ['Ambience', 'Lively', 'Comfortablility', 'Privacy']
    df['experience_score'] = df[experience_cols].mean(axis=1)

    wait_service_cols = ['Order Wait Time', 'Staff Responsivness']
    df['efficiency_score'] = 10 - df[wait_service_cols].mean(axis=1)  # Invert wait time

    # Rating consistency and spread
    all_ratings = df[rating_cols].values
    df['rating_std'] = np.nanstd(all_ratings, axis=1)
    df['rating_range'] = np.nanmax(all_ratings, axis=1) - np.nanmin(all_ratings, axis=1)
    df['rating_consistency'] = 10 - df['rating_std']  # Higher is more consistent

    # Binary features
    binary_cols = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
                   'Dedicated Parking', 'Open Sitting Available']
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    df['total_amenities'] = df[binary_cols].sum(axis=1)
    df['safety_score'] = df['Fire Audit'] + df['Liquor License Obtained']
    df['convenience_score'] = df['Dedicated Parking'] + df['Situated in a Multi Complex']

    # Tier features
    df['Resturant Tier'] = pd.to_numeric(df['Resturant Tier'], errors='coerce').fillna(2)
    df['Restaurant City Tier'] = pd.to_numeric(df['Restaurant City Tier'], errors='coerce').fillna(1)
    df['Restaurant Zomato Rating'] = pd.to_numeric(df['Restaurant Zomato Rating'], errors='coerce').fillna(3)

    # Premium indicators
    df['is_tier1_restaurant'] = (df['Resturant Tier'] == 1).astype(int)
    df['is_tier1_city'] = (df['Restaurant City Tier'] == 1).astype(int)
    df['is_celebrity_endorsed'] = (df['Endorsed By'] == 'Tier A Celebrity').astype(int)
    df['is_high_zomato'] = (df['Restaurant Zomato Rating'] >= 4).astype(int)

    # Power interaction features
    df['tier_celebrity_interaction'] = df['is_tier1_restaurant'] * df['is_celebrity_endorsed']
    df['social_quality_interaction'] = df['social_media_avg'] * df['quality_score']
    df['age_rating_interaction'] = df['restaurant_age'] * df['Overall Restaurant Rating']
    df['location_tier_interaction'] = df['is_business_hub'] * df['is_tier1_city']
    df['premium_experience'] = df['has_premium_cuisine'] * df['experience_score']

    # Advanced mathematical transformations
    df['social_power'] = df['social_media_avg'] ** 0.5
    df['quality_power'] = df['quality_score'] ** 2
    df['age_log_squared'] = df['age_log'] ** 2

    # Ratios and normalized features
    df['rating_per_amenity'] = df['Overall Restaurant Rating'] / (df['total_amenities'] + 1)
    df['social_per_year'] = df['social_media_avg'] / (df['restaurant_age'] + 1)
    df['quality_consistency_ratio'] = df['quality_score'] / (df['rating_std'] + 0.1)

    return df

def apply_target_encoding(df, train_size, target_col='Annual Turnover'):
    """Apply target encoding to categorical features"""

    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()

    target = train_df[target_col]

    # Categories to encode
    cat_cols = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By', 'social_cluster']

    te = TargetEncoder(smoothing=10, min_samples_leaf=5, noise_level=0.01)

    for col in cat_cols:
        if col in df.columns:
            train_encoded, test_encoded = te.encode(train_df[col], target, test_df[col])
            df.loc[:train_size-1, col + '_target_encoded'] = train_encoded
            df.loc[train_size:, col + '_target_encoded'] = test_encoded.values

    return df

def prepare_championship_features(df, train_size):
    """Prepare final championship feature set"""

    # Apply target encoding
    df = apply_target_encoding(df, train_size)

    # Drop non-predictive columns
    drop_cols = ['Registration Number', 'Annual Turnover', 'is_train',
                 'Opening Day of Restaurant', 'Cuisine']

    # Get numeric features only
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Handle any remaining issues
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df['Annual Turnover'][:train_size]

    # Split datasets
    X_train = X[:train_size]
    X_test = X[train_size:]

    print(f"Championship features: {X_train.shape[1]}")
    print(f"Key features: {list(X_train.columns[:15])}")

    return X_train, X_test, y

class ChampionshipEnsemble:
    """Championship-level ensemble with stacking"""

    def __init__(self):
        # Level 1 models (base learners)
        self.level1_models = {
            'rf_deep': RandomForestRegressor(
                n_estimators=1000, max_depth=25, min_samples_split=3,
                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'rf_wide': RandomForestRegressor(
                n_estimators=800, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, max_features=0.7, random_state=123, n_jobs=-1
            ),
            'et_aggressive': ExtraTreesRegressor(
                n_estimators=1200, max_depth=30, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=456, n_jobs=-1
            ),
            'et_balanced': ExtraTreesRegressor(
                n_estimators=800, max_depth=20, min_samples_split=4,
                min_samples_leaf=2, max_features=0.8, random_state=789, n_jobs=-1
            ),
            'gb_deep': GradientBoostingRegressor(
                n_estimators=1000, max_depth=15, learning_rate=0.03,
                subsample=0.8, max_features='sqrt', random_state=101
            ),
            'gb_fast': GradientBoostingRegressor(
                n_estimators=600, max_depth=10, learning_rate=0.08,
                subsample=0.85, max_features=0.7, random_state=202
            )
        }

        # Level 2 model (meta-learner)
        self.level2_model = Ridge(alpha=1.0, random_state=42)

        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.level1_predictions = {}

    def fit(self, X_train, y_train):
        """Fit championship ensemble with stacking"""

        print("🏆 Training Championship Ensemble...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Generate level 1 predictions using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        level1_train_preds = np.zeros((len(X_train), len(self.level1_models)))

        cv_scores = {}

        for i, (name, model) in enumerate(self.level1_models.items()):
            print(f"Training {name}...")

            # Cross-validation predictions for stacking
            fold_preds = np.zeros(len(X_train))
            fold_scores = []

            for train_idx, val_idx in kf.split(X_scaled):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Fit and predict
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                fold_pred = model_copy.predict(X_fold_val)

                fold_preds[val_idx] = fold_pred
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
                fold_scores.append(fold_rmse)

            level1_train_preds[:, i] = fold_preds
            cv_scores[name] = np.mean(fold_scores)
            print(f"{name} CV RMSE: {cv_scores[name]:,.2f}")

            # Fit on full training set for final predictions
            model.fit(X_scaled, y_train)

        # Train level 2 model
        print("Training meta-learner...")
        self.level2_model.fit(level1_train_preds, y_train)

        # Calculate ensemble CV score
        meta_pred = self.level2_model.predict(level1_train_preds)
        ensemble_rmse = np.sqrt(mean_squared_error(y_train, meta_pred))
        print(f"Ensemble CV RMSE: {ensemble_rmse:,.2f}")

        return self

    def predict(self, X_test):
        """Generate championship predictions"""

        X_scaled = self.scaler.transform(X_test)

        # Generate level 1 predictions
        level1_test_preds = np.zeros((len(X_test), len(self.level1_models)))

        for i, (name, model) in enumerate(self.level1_models.items()):
            level1_test_preds[:, i] = model.predict(X_scaled)

        # Generate final prediction using level 2 model
        final_pred = self.level2_model.predict(level1_test_preds)

        return final_pred

def main():
    """Execute championship pipeline"""

    print("🏆 Championship Restaurant Turnover Prediction Starting...")

    # Load and clean data
    df, train_size = load_and_clean_data()

    # Championship feature engineering
    print("🔧 Championship Feature Engineering...")
    df = champion_feature_engineering(df)

    # Prepare features
    print("⚡ Preparing Championship Features...")
    X_train, X_test, y_train = prepare_championship_features(df, train_size)

    # Train championship ensemble
    ensemble = ChampionshipEnsemble()
    ensemble.fit(X_train, y_train)

    # Validation
    print("\n📊 Championship Validation...")
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    val_ensemble = ChampionshipEnsemble()
    val_ensemble.fit(X_train_val, y_train_val)
    val_pred = val_ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"\n🎯 CHAMPIONSHIP RESULTS:")
    print(f"Validation RMSE: {val_rmse:,.2f}")
    print(f"Target to beat: 12,337,503.37")
    improvement = (12337503.37 - val_rmse) / 12337503.37 * 100
    print(f"Improvement: {improvement:+.2f}%")

    # Generate final predictions
    print("\n🚀 Generating Championship Predictions...")
    test_predictions = ensemble.predict(X_test)

    # Create submission
    test_df = pd.read_csv('Test_dataset.csv')
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': test_predictions
    })

    submission_file = 'championship_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n🏆 CHAMPIONSHIP SUBMISSION READY!")
    print(f"📁 File: {submission_file}")
    print(f"📊 Entries: {len(submission)}")
    print(f"🎯 Predicted RMSE: {val_rmse:,.2f}")

    if val_rmse < 12337503.37:
        print("🥇 VICTORY! You should beat the leaderboard!")
    else:
        print("🥈 Close! Consider further optimization")

    return submission

if __name__ == "__main__":
    main()
