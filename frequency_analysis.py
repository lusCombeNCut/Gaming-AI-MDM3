import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def process_patch_notes(patch_notes_dir):
    patch_counts = pd.DataFrame(columns=["appid", "month", "patch_count"])

    for filename in os.listdir(patch_notes_dir):
        if filename.endswith("_patch_notes.json"):
            appid = filename.split("_")[0]  # Extract appid from filename
            file_path = os.path.join(patch_notes_dir, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                patch_data = json.load(file)

            patches = pd.DataFrame(patch_data)
            patches["month"] = pd.to_datetime(patches["date"], unit="s").dt.strftime("%Y-%m")

            # Count patches per month
            patch_count = patches.groupby("month").size().reset_index(name="patch_count")
            patch_count["appid"] = int(appid)  # Convert to integer

            patch_counts = pd.concat([patch_counts, patch_count], ignore_index=True)

    return patch_counts

def merge_data(game_data, patch_counts):
    game_data.rename(columns={"Game_Id": "appid"}, inplace=True)   

    # Ensure 'appid' is integer type in both DataFrames
    game_data["appid"] = game_data["appid"].astype(int)
    patch_counts["appid"] = patch_counts["appid"].astype(int)

    merged = game_data.merge(patch_counts, how="left", on=["appid", "month"])

    # Fill missing patch counts with 0
    merged["patch_count"] = merged["patch_count"].fillna(0)

    merged.to_csv("data_with_patch_counts.csv", index=False)
    print("Merged data saved successfully!")

def load_data(game_data_file, patch_notes_dir):
    data_file = "data_with_patch_counts.csv"

    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}...")
        return pd.read_csv(data_file)

    print(f"{data_file} not found. Processing data...")
    
    game_data = pd.read_csv(game_data_file, encoding="ISO-8859-1")

    # Rename 'Month' to 'month' and format as YYYY-MM
    game_data.rename(columns={"Month": "month", "Game_Id": "appid"}, inplace=True)
    game_data["month"] = pd.to_datetime(game_data["month"], format="%B %Y").dt.strftime("%Y-%m")

    patch_counts = process_patch_notes(patch_notes_dir)
    merge_data(game_data, patch_counts)

    return pd.read_csv(data_file)

def plot_game_trends(df, game_name):
    first_word = game_name.split()[0].lower()  
    game_data = df[df["Game_Name"].str.lower().str.startswith(first_word)].copy()
    
    if game_data.empty:
        print(f"No data found for '{game_name}'")
        return

    # Convert 'month' to categorical format for ordered plotting
    game_data["month"] = pd.Categorical(game_data["month"], ordered=True, categories=sorted(game_data["month"].unique()))

    # Clean and convert '% Gain' column
    game_data["% Gain"] = (
        game_data["% Gain"]
        .astype(str)
        .str.replace("%", "", regex=False)  # Remove % sign
        .str.strip()  # Remove spaces
    )

    # Convert to float (handling errors safely)
    game_data["% Gain"] = pd.to_numeric(game_data["% Gain"], errors="coerce").fillna(0)

    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel("Month", fontsize=16)
    ax1.set_ylabel("% Gain in Avg. Players", color="blue", fontsize=16)
    sns.lineplot(x=game_data["month"], y=game_data["% Gain"], marker="o", color="blue", ax=ax1, label="% Gain", ci=None)
    ax1.tick_params(axis="y", labelcolor="blue")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, fontsize=12, ha="right")
    plt.yticks(fontsize=12)

    # Create second y-axis for Patch Count
    ax2 = ax1.twinx()
    ax2.set_ylabel("Patch Count", color="red", fontsize=16)

    ax2.bar(game_data["month"], game_data["patch_count"], color="red", alpha=0.25)
    ax2.tick_params(axis="y", labelcolor="red")

    # Title & Formatting
    plt.title(f"% Gain vs. Patch Frequency for {game_name}", fontsize=18)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    game_data_file = "C:/Uni/MDM3/gaming/datasets/game_data.csv"
    patch_notes_dir = "C:/Uni/MDM3/gaming/patch_notes"

    merged_data = load_data(game_data_file, patch_notes_dir)
    print("Data loaded successfully!")
    plot_game_trends(merged_data, "Apex Legends")