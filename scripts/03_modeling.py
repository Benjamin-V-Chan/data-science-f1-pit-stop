# - import pandas, sklearn modules, os, joblib
# - define load_data(path)
# - define train_models(X, y): split train/test, scale features, fit LinearRegression and RandomForestRegressor, cross-validate, return fitted models and scores
# - define save_models(models, scores, out_dir): save each model with joblib, write scores to CSV
# - in main:
#     * load processed data
#     * set X = feature cols, y = target (e.g., pit duration)
#     * train_models
#     * save_models