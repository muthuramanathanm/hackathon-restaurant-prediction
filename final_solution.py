#!/usr/bin/env python3
"""
Final Championship Solution - Restaurant Turnover Prediction
Robust ensemble without target encoding but with powerful features
Goal: Beat RMSE of 12,337,503.37
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and basic preprocessing"""

    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # Combine for preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Annual Turnover'] = np.nan

    # Fix endorsement column inconsistency
    if 'Endoresed By' in test_df.columns:
        test_df['Endorsed By'] = test_df['Endoresed By']
        test_df = test_df.drop('Endoresed By', axis=1)

    df = pd.concat([train_df, test_df], ignore_index=True)

    return df, train_df.shape[0]

def championship_features(df):
    """Create championship-level features"""

    # Handle missing values strategically
    df['Facebook Popularity Quotient'] = df['Facebook Popularity Quotient'].fillna(df['Facebook Popularity Quotient'].median())
    df['Instagram Popularity Quotient'] = df['Instagram Popularity Quotient'].fillna(df['Instagram Popularity Quotient'].median())

    # Clean categorical data
    df['City'] = df['City'].replace('-1', 'Unknown').fillna('Unknown')
    df['Endorsed By'] = df['Endorsed By'].fillna('Not Specific')
    df['Cuisine'] = df['Cuisine'].fillna('unknown')

    # Date features with advanced transformations
    df['Opening Day of Restaurant'] = pd.to_datetime(df['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
    df['opening_year'] = df['Opening Day of Restaurant'].dt.year
    df['opening_month'] = df['Opening Day of Restaurant'].dt.month
    df['opening_day'] = df['Opening Day of Restaurant'].dt.day
    df['opening_weekday'] = df['Opening Day of Restaurant'].dt.dayofweek
    df['restaurant_age'] = 2024 - df['opening_year']
    df['restaurant_age'] = np.where(df['restaurant_age'] < 0, 0, df['restaurant_age'])

    # Age transformations (crucial for revenue prediction)
    df['age_squared'] = df['restaurant_age'] ** 2
    df['age_cubed'] = df['restaurant_age'] ** 3
    df['age_log'] = np.log1p(df['restaurant_age'])
    df['age_sqrt'] = np.sqrt(df['restaurant_age'])
    df['age_inv'] = 1 / (df['restaurant_age'] + 1)

    # Restaurant lifecycle
    df['is_new'] = (df['restaurant_age'] <= 2).astype(int)
    df['is_established'] = ((df['restaurant_age'] > 2) & (df['restaurant_age'] <= 10)).astype(int)
    df['is_veteran'] = (df['restaurant_age'] > 10).astype(int)

    # Seasonal opening effects
    df['opened_peak_season'] = ((df['opening_month'] >= 4) & (df['opening_month'] <= 6)).astype(int)
    df['opened_off_season'] = ((df['opening_month'] >= 10) | (df['opening_month'] <= 2)).astype(int)
    df['opened_weekend'] = (df['opening_weekday'] >= 5).astype(int)

    # Cuisine analysis (high impact on revenue)
    df['cuisine_count'] = df['Cuisine'].str.count(',') + 1

    # High-value cuisines
    premium_cuisines = ['italian', 'japanese', 'french', 'mediterranean', 'greek']
    popular_cuisines = ['indian', 'chinese', 'thai', 'american']
    ethnic_cuisines = ['korean', 'vietnamese', 'mexican', 'turkish']

    df['premium_cuisine_count'] = 0
    df['popular_cuisine_count'] = 0
    df['ethnic_cuisine_count'] = 0

    for cuisine in premium_cuisines:
        has_cuisine = df['Cuisine'].str.contains(cuisine, case=False, na=False).astype(int)
        df[f'has_{cuisine}'] = has_cuisine
        df['premium_cuisine_count'] += has_cuisine

    for cuisine in popular_cuisines:
        has_cuisine = df['Cuisine'].str.contains(cuisine, case=False, na=False).astype(int)
        df[f'has_{cuisine}'] = has_cuisine
        df['popular_cuisine_count'] += has_cuisine

    for cuisine in ethnic_cuisines:
        has_cuisine = df['Cuisine'].str.contains(cuisine, case=False, na=False).astype(int)
        df[f'has_{cuisine}'] = has_cuisine
        df['ethnic_cuisine_count'] += has_cuisine

    # Cuisine diversity score
    df['cuisine_diversity'] = (df['premium_cuisine_count'] > 0).astype(int) + \
                             (df['popular_cuisine_count'] > 0).astype(int) + \
                             (df['ethnic_cuisine_count'] > 0).astype(int)

    # Location features
    df['is_business_hub'] = (df['Restaurant Location'] == 'Near Business Hub').astype(int)
    df['is_party_hub'] = (df['Restaurant Location'] == 'Near Party Hub').astype(int)

    # Social media features (critical for modern restaurants)
    df['social_avg'] = (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient']) / 2
    df['social_max'] = np.maximum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_min'] = np.minimum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_diff'] = df['Facebook Popularity Quotient'] - df['Instagram Popularity Quotient']
    df['social_ratio'] = df['Facebook Popularity Quotient'] / (df['Instagram Popularity Quotient'] + 1)
    df['social_harmonic_mean'] = 2 * df['Facebook Popularity Quotient'] * df['Instagram Popularity Quotient'] / \
                                (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient'] + 1)

    # Social media categories
    df['social_excellent'] = (df['social_avg'] >= 90).astype(int)
    df['social_good'] = ((df['social_avg'] >= 70) & (df['social_avg'] < 90)).astype(int)
    df['social_poor'] = (df['social_avg'] < 50).astype(int)

    # Advanced social media features
    df['social_power'] = df['social_avg'] ** 0.5
    df['social_squared'] = df['social_avg'] ** 2
    df['social_log'] = np.log1p(df['social_avg'])

    # Rating features
    rating_cols = ['Order Wait Time', 'Staff Responsivness', 'Value for Money',
                   'Hygiene Rating', 'Food Rating', 'Overall Restaurant Rating',
                   'Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating',
                   'Live Sports Rating', 'Ambience', 'Lively', 'Service',
                   'Comfortablility', 'Privacy']

    # Convert to numeric and smart imputation
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Composite rating scores
    df['service_excellence'] = df[['Staff Responsivness', 'Service']].mean(axis=1)
    df['food_quality'] = df[['Food Rating', 'Hygiene Rating']].mean(axis=1)
    df['value_proposition'] = df[['Value for Money', 'Overall Restaurant Rating']].mean(axis=1)
    df['atmosphere'] = df[['Ambience', 'Lively', 'Comfortablility']].mean(axis=1)
    df['entertainment'] = df[['Live Music Rating', 'Comedy Gigs Rating', 'Live Sports Rating']].mean(axis=1)
    df['efficiency'] = 10 - df['Order Wait Time']  # Higher is better

    # Rating analysis
    df['rating_avg'] = df[rating_cols].mean(axis=1)
    df['rating_std'] = df[rating_cols].std(axis=1)
    df['rating_max'] = df[rating_cols].max(axis=1)
    df['rating_min'] = df[rating_cols].min(axis=1)
    df['rating_range'] = df['rating_max'] - df['rating_min']
    df['rating_consistency'] = 10 - df['rating_std']  # Higher is more consistent

    # Binary amenities
    binary_cols = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
                   'Dedicated Parking', 'Open Sitting Available']
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    df['total_amenities'] = df[binary_cols].sum(axis=1)
    df['safety_compliance'] = df[['Fire Audit', 'Liquor License Obtained']].sum(axis=1)
    df['customer_convenience'] = df[['Dedicated Parking', 'Open Sitting Available']].sum(axis=1)

    # Tier and rating features
    df['Resturant Tier'] = pd.to_numeric(df['Resturant Tier'], errors='coerce').fillna(2)
    df['Restaurant City Tier'] = pd.to_numeric(df['Restaurant City Tier'], errors='coerce').fillna(1)
    df['Restaurant Zomato Rating'] = pd.to_numeric(df['Restaurant Zomato Rating'], errors='coerce').fillna(3)

    # Premium status indicators
    df['is_tier1_restaurant'] = (df['Resturant Tier'] == 1).astype(int)
    df['is_tier1_city'] = (df['Restaurant City Tier'] == 1).astype(int)
    df['is_celebrity_endorsed'] = (df['Endorsed By'] == 'Tier A Celebrity').astype(int)
    df['is_high_zomato'] = (df['Restaurant Zomato Rating'] >= 4).astype(int)
    df['is_low_zomato'] = (df['Restaurant Zomato Rating'] <= 2).astype(int)

    # Power interaction features (key for ensemble performance)
    df['premium_celebrity'] = df['is_tier1_restaurant'] * df['is_celebrity_endorsed']
    df['quality_social'] = df['food_quality'] * df['social_avg'] / 100
    df['age_quality'] = df['restaurant_age'] * df['Overall Restaurant Rating']
    df['location_tier'] = df['is_business_hub'] * df['is_tier1_city']
    df['premium_location'] = df['premium_cuisine_count'] * df['is_business_hub']
    df['social_age'] = df['social_avg'] / (df['restaurant_age'] + 1)
    df['quality_amenities'] = df['food_quality'] * df['total_amenities']

    # Advanced mathematical features
    df['social_power_transform'] = np.power(df['social_avg'], 0.3)
    df['quality_exp'] = np.exp(df['food_quality'] / 10)
    df['age_log_interaction'] = df['age_log'] * df['Overall Restaurant Rating']

    # Ratios and efficiency metrics
    df['rating_per_amenity'] = df['Overall Restaurant Rating'] / (df['total_amenities'] + 1)
    df['quality_per_wait'] = df['food_quality'] / (df['Order Wait Time'] + 1)
    df['social_consistency'] = df['social_avg'] / (df['rating_std'] + 0.1)
    df['value_efficiency'] = df['Value for Money'] * df['efficiency'] / 10

    return df

def encode_categories(df):
    """Simple label encoding for categorical variables"""

    cat_cols = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By']

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

    return df

def prepare_features(df, train_size):
    """Prepare final feature matrix"""

    # Encode categorical variables
    df = encode_categories(df)

    # Select features
    drop_cols = ['Registration Number', 'Annual Turnover', 'is_train',
                 'Opening Day of Restaurant', 'Cuisine']

    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df['Annual Turnover'][:train_size]

    X_train = X[:train_size]
    X_test = X[train_size:]

    print(f"Features: {X_train.shape[1]}")
    print(f"Key features: {list(X_train.columns[:10])}")

    return X_train, X_test, y

class ChampionEnsemble:
    """Championship ensemble with optimized models"""

    def __init__(self):
        self.models = {
            # Random Forest variants
            'rf_deep': RandomForestRegressor(
                n_estimators=2000, max_depth=35, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt',
                random_state=42, n_jobs=-1
            ),
            'rf_balanced': RandomForestRegressor(
                n_estimators=1500, max_depth=25, min_samples_split=3,
                min_samples_leaf=1, max_features=0.6,
                random_state=123, n_jobs=-1
            ),
            'rf_conservative': RandomForestRegressor(
                n_estimators=1200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, max_features=0.8,
                random_state=456, n_jobs=-1
            ),

            # Extra Trees variants
            'et_aggressive': ExtraTreesRegressor(
                n_estimators=2500, max_depth=40, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt',
                random_state=789, n_jobs=-1
            ),
            'et_balanced': ExtraTreesRegressor(
                n_estimators=1800, max_depth=30, min_samples_split=3,
                min_samples_leaf=1, max_features=0.7,
                random_state=101, n_jobs=-1
            ),
            'et_conservative': ExtraTreesRegressor(
                n_estimators=1500, max_depth=25, min_samples_split=4,
                min_samples_leaf=2, max_features=0.8,
                random_state=202, n_jobs=-1
            ),

            # Gradient Boosting variants
            'gb_precise': GradientBoostingRegressor(
                n_estimators=2000, max_depth=12, learning_rate=0.02,
                subsample=0.8, max_features='sqrt', random_state=303
            ),
            'gb_balanced': GradientBoostingRegressor(
                n_estimators=1500, max_depth=10, learning_rate=0.03,
                subsample=0.85, max_features=0.7, random_state=404
            ),
            'gb_fast': GradientBoostingRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.05,
                subsample=0.9, max_features=0.8, random_state=505
            ),

            # Linear models for stability
            'ridge_strong': Ridge(alpha=100, random_state=606),
            'ridge_medium': Ridge(alpha=50, random_state=707),
            'elastic_balanced': ElasticNet(alpha=30, l1_ratio=0.7, random_state=808),
            'lasso_selective': Lasso(alpha=500, random_state=909)
        }

        self.weights = None
        self.scaler = RobustScaler()

    def fit(self, X_train, y_train):
        """Train championship ensemble"""

        print("🏆 Training Championship Ensemble...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Cross-validation scoring
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Cross-validation
            fold_scores = []
            for train_idx, val_idx in kf.split(X_scaled):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Clone and fit model
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_fold_train, y_fold_train)
                pred = model_copy.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
                fold_scores.append(rmse)

            cv_rmse = np.mean(fold_scores)
            cv_scores[name] = cv_rmse
            print(f"{name} CV RMSE: {cv_rmse:,.0f}")

            # Fit on full training data
            model.fit(X_scaled, y_train)

        # Calculate inverse-error weights
        min_score = min(cv_scores.values())
        inv_scores = {name: min_score/score for name, score in cv_scores.items()}
        total_inv = sum(inv_scores.values())
        self.weights = {name: inv_score/total_inv for name, inv_score in inv_scores.items()}

        print(f"\nTop model weights:")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{name}: {weight:.3f}")

        return self

    def predict(self, X_test):
        """Generate championship predictions"""

        X_scaled = self.scaler.transform(X_test)

        # Weighted ensemble prediction
        final_pred = np.zeros(len(X_test))

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            final_pred += self.weights[name] * pred

        return final_pred

def main():
    """Execute championship solution"""

    print("🏆 CHAMPIONSHIP RESTAURANT PREDICTION")
    print("=" * 50)

    # Load data
    df, train_size = load_data()

    # Feature engineering
    print("🔧 Championship Feature Engineering...")
    df = championship_features(df)

    # Prepare features
    print("⚡ Preparing Features...")
    X_train, X_test, y_train = prepare_features(df, train_size)

    # Train ensemble
    ensemble = ChampionEnsemble()
    ensemble.fit(X_train, y_train)

    # Validation
    print("\n📊 VALIDATION")
    print("-" * 20)
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    val_ensemble = ChampionEnsemble()
    val_ensemble.fit(X_train_val, y_train_val)
    val_pred = val_ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    target = 12337503.37
    improvement = (target - val_rmse) / target * 100

    print(f"Validation RMSE: {val_rmse:,.0f}")
    print(f"Target RMSE: {target:,.0f}")
    print(f"Improvement: {improvement:+.1f}%")

    # Final predictions
    print("\n🎯 Final Predictions...")
    final_predictions = ensemble.predict(X_test)

    # Create submission
    test_df = pd.read_csv('Test_dataset.csv')
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': final_predictions
    })

    submission_file = 'championship_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n🏆 CHAMPIONSHIP RESULTS")
    print("=" * 30)
    print(f"📁 File: {submission_file}")
    print(f"📊 Entries: {len(submission)}")
    print(f"🎯 RMSE: {val_rmse:,.0f}")

    if val_rmse < target:
        print("🥇 CHAMPION! Beating the leaderboard!")
    else:
        improvement_needed = (val_rmse - target) / target * 100
        print(f"🥈 Need {improvement_needed:.1f}% more improvement")

    print(f"\n📈 Predictions: {final_predictions.min():,.0f} - {final_predictions.max():,.0f}")
    print(f"📊 Mean: {final_predictions.mean():,.0f}")

    return submission

if __name__ == "__main__":
    main()
