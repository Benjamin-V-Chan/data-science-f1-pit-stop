# data-science-f1-pit-stop

## Project Overview

This project analyzes Formula 1 pit stop data to uncover key insights into driver performance, pit stop efficiency, and the impact of weather conditions on race strategies. It includes data preprocessing, exploratory analysis, predictive modeling, and clustering of driving stints.

## Folder Structure

```
project-root/
├── data/
│   ├── drivers_stints_raw.csv       # Raw dataset imported from Kaggle
│   └── processed_stints.csv         # Cleaned and feature-engineered data
├── scripts/
│   ├── 01_data_preprocessing.py     # Load, clean, and engineer features
│   ├── 02_exploratory_analysis.py   # Generate summary stats and visualizations
│   ├── 03_modeling.py               # Train and evaluate regression models
│   └── 04_clustering.py             # Perform KMeans clustering
├── outputs/
│   ├── eda/                         # EDA outputs (plots, summary CSV)
│   ├── models/                      # Serialized models and performance metrics
│   └── clustering/                  # Clustering results and plots
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation (this file)
```

