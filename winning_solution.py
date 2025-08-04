#!/usr/bin/env python3
"""
Restaurant Annual Turnover Prediction - Winning Solution
Goal: Beat RMSE of 12,337,503.37
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the training and test datasets"""

    # Load data
    train_df = pd.read_csv('Train_dataset.csv')
    test_df = pd.read_csv('Test_dataset.csv')

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Combine for preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['Annual Turnover'] = np.nan

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    return combined_df, train_df.shape[0]

def advanced_feature_engineering(df):
    """Create powerful predictive features"""

    # Handle missing values strategically
    df['Facebook Popularity Quotient'] = df['Facebook Popularity Quotient'].fillna(df['Facebook Popularity Quotient'].median())
    df['Instagram Popularity Quotient'] = df['Instagram Popularity Quotient'].fillna(df['Instagram Popularity Quotient'].median())

    # City cleaning - handle -1 and missing
    df['City'] = df['City'].replace('-1', 'Unknown')
    df['City'] = df['City'].fillna('Unknown')

    # Parse opening date and extract valuable time features
    df['Opening Day of Restaurant'] = pd.to_datetime(df['Opening Day of Restaurant'], format='%d-%m-%Y', errors='coerce')
    df['opening_year'] = df['Opening Day of Restaurant'].dt.year
    df['opening_month'] = df['Opening Day of Restaurant'].dt.month
    df['opening_day'] = df['Opening Day of Restaurant'].dt.day
    df['restaurant_age'] = 2024 - df['opening_year']
    df['restaurant_age'] = np.where(df['restaurant_age'] < 0, 0, df['restaurant_age'])

    # Season effects
    df['opened_in_summer'] = ((df['opening_month'] >= 4) & (df['opening_month'] <= 6)).astype(int)
    df['opened_in_winter'] = ((df['opening_month'] >= 10) | (df['opening_month'] <= 2)).astype(int)

    # Handle endorsement inconsistency between train/test
    df['Endorsed By'] = df['Endorsed By'].fillna('Not Specific')
    if 'Endoresed By' in df.columns:
        df['Endorsed By'] = df['Endorsed By'].fillna(df['Endoresed By'])

    # Advanced cuisine analysis
    df['cuisine_count'] = df['Cuisine'].str.count(',') + 1
    df['has_indian'] = df['Cuisine'].str.contains('indian', case=False, na=False).astype(int)
    df['has_italian'] = df['Cuisine'].str.contains('italian', case=False, na=False).astype(int)
    df['has_chinese'] = df['Cuisine'].str.contains('chinese', case=False, na=False).astype(int)
    df['has_greek'] = df['Cuisine'].str.contains('greek', case=False, na=False).astype(int)
    df['has_thai'] = df['Cuisine'].str.contains('thai', case=False, na=False).astype(int)

    # Premium cuisine indicator
    premium_cuisines = ['japanese', 'french', 'italian']
    df['has_premium_cuisine'] = df['Cuisine'].str.lower().str.contains('|'.join(premium_cuisines), na=False).astype(int)

    # Location insights
    df['is_business_hub'] = (df['Restaurant Location'] == 'Near Business Hub').astype(int)
    df['is_party_hub'] = (df['Restaurant Location'] == 'Near Party Hub').astype(int)

    # Social media powerhouse features
    df['social_media_avg'] = (df['Facebook Popularity Quotient'] + df['Instagram Popularity Quotient']) / 2
    df['social_media_max'] = np.maximum(df['Facebook Popularity Quotient'], df['Instagram Popularity Quotient'])
    df['social_media_diff'] = df['Facebook Popularity Quotient'] - df['Instagram Popularity Quotient']
    df['is_social_media_star'] = (df['social_media_avg'] > 90).astype(int)

    # Rating features with smart imputation
    rating_cols = ['Order Wait Time', 'Staff Responsivness', 'Value for Money',
                   'Hygiene Rating', 'Food Rating', 'Overall Restaurant Rating',
                   'Live Music Rating', 'Comedy Gigs Rating', 'Value Deals Rating',
                   'Live Sports Rating', 'Ambience', 'Lively', 'Service',
                   'Comfortablility', 'Privacy']

    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Create powerful composite ratings
    service_cols = ['Staff Responsivness', 'Service', 'Order Wait Time']
    df['service_excellence'] = df[service_cols].mean(axis=1)

    quality_cols = ['Food Rating', 'Value for Money', 'Hygiene Rating']
    df['quality_score'] = df[quality_cols].mean(axis=1)

    experience_cols = ['Ambience', 'Lively', 'Comfortablility', 'Privacy']
    df['experience_score'] = df[experience_cols].mean(axis=1)

    entertainment_cols = ['Live Music Rating', 'Comedy Gigs Rating', 'Live Sports Rating']
    df['entertainment_value'] = df[entertainment_cols].mean(axis=1)

    # Binary amenities
    binary_cols = ['Fire Audit', 'Liquor License Obtained', 'Situated in a Multi Complex',
                   'Dedicated Parking', 'Open Sitting Available']
    for col in binary_cols:
        df[col] = df[col].fillna(0).astype(int)

    df['total_amenities'] = df[binary_cols].sum(axis=1)

    # Tier analysis
    df['Resturant Tier'] = pd.to_numeric(df['Resturant Tier'], errors='coerce').fillna(2)
    df['Restaurant City Tier'] = pd.to_numeric(df['Restaurant City Tier'], errors='coerce').fillna(1)

    # Premium indicators
    df['is_tier1_restaurant'] = (df['Resturant Tier'] == 1).astype(int)
    df['is_tier1_city'] = (df['Restaurant City Tier'] == 1).astype(int)
    df['is_celebrity_endorsed'] = (df['Endorsed By'] == 'Tier A Celebrity').astype(int)

    # Power interaction features
    df['tier_celebrity_combo'] = df['is_tier1_restaurant'] * df['is_celebrity_endorsed']
    df['social_quality_interaction'] = df['social_media_avg'] * df['quality_score']
    df['age_rating_synergy'] = df['restaurant_age'] * df['Overall Restaurant Rating']
    df['location_tier_combo'] = df['is_business_hub'] * df['is_tier1_city']

    # Advanced ratios
    df['rating_consistency'] = df['Overall Restaurant Rating'] / (df['Food Rating'] + 0.1)
    df['social_per_year'] = df['social_media_avg'] / (df['restaurant_age'] + 1)

    return df

def encode_categorical_features(df):
    """Smart categorical encoding"""

    # High-impact categorical features
    categorical_cols = ['City', 'Restaurant Type', 'Restaurant Theme', 'Endorsed By']

    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders

def prepare_features(df, train_size):
    """Prepare optimized feature set"""

    # Strategic feature selection
    drop_cols = ['Registration Number', 'Annual Turnover', 'is_train',
                 'Opening Day of Restaurant', 'Cuisine']

    # Get all numeric features
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])

    # Handle any remaining NaNs or infinities
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df['Annual Turnover'][:train_size]

    # Split datasets
    X_train = X[:train_size]
    X_test = X[train_size:]

    print(f"Final features: {X_train.shape[1]}")
    print(f"Sample features: {list(X_train.columns[:10])}")

    return X_train, X_test, y

class PowerEnsemble:
    """High-performance ensemble optimized for this problem"""

    def __init__(self):
        self.models = {
            'rf_deep': RandomForestRegressor(
                n_estimators=800,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'rf_wide': RandomForestRegressor(
                n_estimators=600,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=123,
                n_jobs=-1
            ),
            'et_aggressive': ExtraTreesRegressor(
                n_estimators=700,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=456,
                n_jobs=-1
            ),
            'gb_optimized': GradientBoostingRegressor(
                n_estimators=800,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.85,
                max_features='sqrt',
                random_state=789
            ),
            'ridge_strong': Ridge(alpha=100, random_state=42),
            'elastic_balanced': ElasticNet(alpha=50, l1_ratio=0.7, random_state=42)
        }

        self.weights = None
        self.scaler = RobustScaler()

    def fit(self, X_train, y_train):
        """Train ensemble with dynamic weighting"""

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Train and validate each model
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"Training {name}...")

            # Cross-validation
            scores = cross_val_score(model, X_scaled, y_train,
                                   cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-scores.mean())
            cv_scores[name] = cv_rmse
            print(f"{name} CV RMSE: {cv_rmse:,.2f}")

            # Fit on full dataset
            model.fit(X_scaled, y_train)

        # Calculate inverse-error weights
        inv_scores = {name: 1/score for name, score in cv_scores.items()}
        total_inv = sum(inv_scores.values())
        self.weights = {name: inv_score/total_inv for name, inv_score in inv_scores.items()}

        print("\nOptimal ensemble weights:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.3f}")

        return self

    def predict(self, X_test):
        """Generate ensemble predictions"""

        X_scaled = self.scaler.transform(X_test)

        ensemble_pred = np.zeros(len(X_test))

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            ensemble_pred += self.weights[name] * pred

        return ensemble_pred

def main():
    """Execute winning pipeline"""

    print("🚀 Loading Restaurant Turnover Prediction Data...")
    df, train_size = load_and_preprocess_data()

    print("🔧 Advanced Feature Engineering...")
    df = advanced_feature_engineering(df)

    print("📊 Encoding Categorical Features...")
    df, encoders = encode_categorical_features(df)

    print("⚡ Preparing Final Features...")
    X_train, X_test, y_train = prepare_features(df, train_size)

    print("🎯 Training Power Ensemble...")
    ensemble = PowerEnsemble()
    ensemble.fit(X_train, y_train)

    # Validation check
    print("\n📈 Validation Analysis...")
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    val_ensemble = PowerEnsemble()
    val_ensemble.fit(X_train_val, y_train_val)
    val_pred = val_ensemble.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"Validation RMSE: {val_rmse:,.2f}")
    print(f"Target to beat: 12,337,503.37")
    print(f"Improvement: {((12337503.37 - val_rmse) / 12337503.37 * 100):+.2f}%")

    # Final predictions
    print("\n🏆 Generating Final Predictions...")
    test_predictions = ensemble.predict(X_test)

    # Create submission
    test_df = pd.read_csv('Test_dataset.csv')
    submission = pd.DataFrame({
        'Registration Number': test_df['Registration Number'],
        'Annual Turnover': test_predictions
    })

    # Ensure exactly 500 entries as required
    assert len(submission) == 500, f"Expected 500 entries, got {len(submission)}"

    submission_file = 'winning_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"\n✅ SUCCESS! Submission saved to {submission_file}")
    print(f"📁 Entries: {len(submission)}")
    print(f"📊 Predicted RMSE: {val_rmse:,.2f}")
    print(f"🎯 Ready to beat the leaderboard!")

    return submission

if __name__ == "__main__":
    main()
