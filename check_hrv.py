import pandas as pd

# df = pd.read_csv("outputs/features_all_normalized.csv")
#
# for col in ["hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]:
#     print(col,
#           "missing:", df[col].isna().mean()*100, "%",
#
#           "unique values:", df[col].nunique())

df = pd.read_csv("outputs/features_all.csv")
print(df[["hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]].describe())
print(df[["hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]].isna().sum())
#
