import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed(path):
    return pd.read_csv(path)

def summary_stats(df, out_dir):
    stats = df.describe()
    stats.to_csv(os.path.join(out_dir, 'summary_statistics.csv'))

def plot_distributions(df, cols, out_dir):
    for col in cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(out_dir, f'{col}_distribution.png'))
        plt.close()

def plot_correlations(df, cols, out_dir):
    corr = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(out_dir, 'correlation_matrix.png'))
    plt.close()

def main():
    proc_path = os.path.join('data', 'processed_stints.csv')
    out_dir = os.path.join('outputs', 'eda')
    os.makedirs(out_dir, exist_ok=True)
    df = load_processed(proc_path)
    summary_stats(df, out_dir)
    numeric_cols = ['Aggression score', 'Fast lap attempts', 'Tire usage aggression',
                    'Number of pit stops', 'average stop time', 'pit duration',
                    'Track temperature', 'air temperature', 'humidity', 'wind speed']
    plot_distributions(df, numeric_cols, out_dir)
    plot_correlations(df, numeric_cols, out_dir)

if __name__ == "__main__":
    main()
