import pandas as pd
df = pd.read_csv("data/haryana_nasa_daily.csv")
df.to_csv("data/processed/haryana_processed.csv", index=False)
