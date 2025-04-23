import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data(path):
    return pd.read_csv(path)

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2')
    rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')

    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    return {
        'scaler': scaler,
        'LinearRegression': (lr, lr_scores.mean()),
        'RandomForest': (rf, rf_scores.mean())
    }

def save_models(models, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name, (model, score) in models.items():
        if name != 'scaler':
            joblib.dump(model, os.path.join(out_dir, f'{name}.joblib'))
            results.append({'model': name, 'cv_r2': score})
    joblib.dump(models['scaler'], os.path.join(out_dir, 'scaler.joblib'))
    df_scores = pd.DataFrame(results)
    df_scores.to_csv(os.path.join(out_dir, 'model_performance.csv'), index=False)

def main():
    proc_path = os.path.join('data', 'processed_stints.csv')
    out_dir = os.path.join('outputs', 'models')
    df = load_data(proc_path)
    feature_cols = ['Aggression score', 'Fast lap attempts', 'Tire usage aggression',
                    'Number of pit stops', 'stint_length',
                    'Track temperature', 'air temperature', 'humidity', 'wind speed']
    X = df[feature_cols]
    y = df['pit duration']
    models = train_models(X, y)
    save_models(models, out_dir)

if __name__ == "__main__":
    main()
