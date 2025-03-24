import pandas as pd

# Read the CSV file
df = pd.read_csv("datasets/game_data.csv")

# Replace any occurrence of "Last 30 Days" (case-insensitive) in the Month column with "Mar-25"
df["Month"] = df["Month"].replace(r'(?i)last 30 days', "March 2025", regex=True)

# Save the updated DataFrame back to game_data.csv
df.to_csv("game_data.csv", index=False)

print("Updated game_data.csv successfully.")
