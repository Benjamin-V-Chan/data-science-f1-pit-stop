import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    return pd.read_csv(path)

def perform_clustering(X, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return labels, score

def plot_clusters(X, labels, out_dir, k):
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    df_plot = pd.DataFrame({'PC1': comps[:,0], 'PC2': comps[:,1], 'Cluster': labels})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='tab10')
    plt.title(f'KMeans Clusters (k={k})')
    plt.savefig(os.path.join(out_dir, f'clusters_k{k}.png'))
    plt.close()

def main():
    proc_path = os.path.join('data', 'processed_stints.csv')
    out_dir = os.path.join('outputs', 'clustering')
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(proc_path)
    metrics = ['Aggression score', 'Fast lap attempts', 'Tire usage aggression',
               'Number of pit stops', 'stint_length']
    X = df[metrics]
    silhouette_scores = []
    for k in range(2, 7):
        labels, score = perform_clustering(X, k)
        silhouette_scores.append({'k': k, 'silhouette': score})
    pd.DataFrame(silhouette_scores).to_csv(os.path.join(out_dir, 'silhouette_scores.csv'), index=False)
    best_k = max(silhouette_scores, key=lambda x: x['silhouette'])['k']
    labels, _ = perform_clustering(X, best_k)
    plot_clusters(X, labels, out_dir, best_k)

if __name__ == "__main__":
    main()
