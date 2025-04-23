import os
import pandas as pd

def load_raw(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()
    df['Aggression score'] = df['Aggression score'].fillna(df['Aggression score'].median())
    df = df.dropna(subset=['Driver', 'Race Name', 'Season'])
    return df

def feature_engineering(df):
    df['stint_length'] = df['stint length']
    numeric_cols = ['Aggression score', 'Fast lap attempts', 'Tire usage aggression',
                    'Number of pit stops', 'average stop time', 'pit duration',
                    'Track temperature', 'air temperature', 'humidity', 'wind speed']
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df

def main():
    raw_path = os.path.join('data', 'drivers_stints_raw.csv')
    out_path = os.path.join('data', 'processed_stints.csv')
    df = load_raw(raw_path)
    df_clean = clean_data(df)
    df_feat = feature_engineering(df_clean)
    df_feat.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()