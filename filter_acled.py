import pandas as pd

# Load the full ACLED CSV
df_acled = pd.read_csv("data/ACLED data 2018 to 2025.csv")

# Keep only Ukraine events
df_ukraine = df_acled[df_acled["country"] == "Ukraine"]

# Convert event_date to datetime for filtering
df_ukraine["event_date"] = pd.to_datetime(df_ukraine["event_date"], format="%Y-%m-%d")

# Filter to events on or after Feb 1, 2022
start_date = pd.Timestamp("2022-02-01")
df_ukraine_recent = df_ukraine[df_ukraine["event_date"] >= start_date]

# Keep only relevant columns
columns_to_keep = [
    "event_id_cnty",
    "event_date",
    "event_type",
    "actor1",
    "actor2",
    "interaction",
    "country",
    "admin1",
    "admin2",
    "latitude",
    "longitude",
    "fatalities"
]
df_ukraine_filtered = df_ukraine_recent[columns_to_keep]

# Write out the smaller CSV
df_ukraine_filtered.to_csv("data/acled_ukraine_filtered.csv", index=False)

# Print size reduction for confirmation
orig_size = df_acled.memory_usage(deep=True).sum() / (1024 * 1024)
filt_size = df_ukraine_filtered.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Original ~{orig_size:.1f} MB; Filtered ~{filt_size:.1f} MB")

