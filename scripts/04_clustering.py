# - import pandas, sklearn KMeans, silhouette_score, matplotlib, seaborn, os
# - define load_data(path)
# - define perform_clustering(df, cols, n_clusters): scale data, fit KMeans, compute silhouette, return labels and score
# - define plot_clusters(df, cols, labels, out_dir): scatter first two principal components colored by cluster
# - in main:
#     * load processed data
#     * choose metrics for clustering
#     * run perform_clustering for a range of k to find best k
#     * save silhouette scores to CSV
#     * plot clusters for best k