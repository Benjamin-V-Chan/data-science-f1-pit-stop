# - import pandas and numpy
# - define load_raw(path) to read raw CSV
# - define clean_data(df):
#     * drop duplicates
#     * handle missing values (e.g., fill or drop)
#     * convert datatypes as needed
# - define feature_engineering(df):
#     * compute stint_duration = pit_stop_lap_end - pit_stop_lap_start or similar
#     * normalize or scale selected numeric columns
# - in main:
#     * load raw from 'data/drivers_stints_raw.csv'
#     * clean and engineer features
#     * save processed to 'data/processed_stints.csv'