import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def process_patch_notes(patch_notes_dir):
    patch_intervals = pd.DataFrame(columns=["appid", "month", "prev_gap", "prev_prev_gap"])

    for filename in os.listdir(patch_notes_dir):
        if filename.endswith("_patch_notes.json"):
            appid = filename.split("_")[0]  # Extract appid from filename
            file_path = os.path.join(patch_notes_dir, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                patch_data = json.load(file)

            patches = pd.DataFrame(patch_data)
            patches["date"] = pd.to_datetime(patches["date"], unit="s")
            patches = patches.sort_values("date")  # Ensure chronological order
            
            patches["month"] = patches["date"].dt.strftime("%Y-%m")
            patches["prev_gap"] = patches["date"].diff().dt.days
            patches["prev_prev_gap"] = patches["prev_gap"].shift(1)
            
            patch_intervals = pd.concat([
                patch_intervals,
                patches[["month", "prev_gap", "prev_prev_gap"]].assign(appid=int(appid))
            ], ignore_index=True)
    
    return patch_intervals

def merge_data(game_data, patch_intervals):
    game_data.rename(columns={"Game_Id": "appid"}, inplace=True)
    game_data["appid"] = game_data["appid"].astype(int)
    patch_intervals["appid"] = patch_intervals["appid"].astype(int)
    
    merged = game_data.merge(patch_intervals, how="left", on=["appid", "month"])
    
    merged.to_csv("data_with_patch_intervals.csv", index=False)
    print("Merged data saved successfully!")

def load_data(game_data_file, patch_notes_dir):
    data_file = "data_with_patch_intervals.csv"
    
    if os.path.exists(data_file):
        print(f"Loading existing data from {data_file}...")
        return pd.read_csv(data_file)
    
    print(f"{data_file} not found. Processing data...")
    
    game_data = pd.read_csv(game_data_file, encoding="ISO-8859-1")
    game_data.rename(columns={"Month": "month", "Game_Id": "appid"}, inplace=True)
    game_data["month"] = pd.to_datetime(game_data["month"], format="%B %Y").dt.strftime("%Y-%m")
    
    patch_intervals = process_patch_notes(patch_notes_dir)
    merge_data(game_data, patch_intervals)
    
    return pd.read_csv(data_file)



def plot_correlation_heatmap(df):
    print(df.columns)
    """ Heatmap showing correlation between patch intervals and player count. """
    cols = [ "Avg. Players", "Gain", "prev_gap", "prev_prev_gap"]
    
    # Apply pd.to_numeric to ensure the columns are numeric
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # Compute the correlation matrix
    correlation_matrix = df[cols].corr()

    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = ["Avg. # Players", "Gain #", "Patch Gap", "Prev. Patch Gap"]
    correlation_matrix.index = ["Avg. # Players", "Gain #", "Patch Gap", "Prev. Patch Gap"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


def plot_correlation_heatmap_specific(df, game_name):
    """ Heatmap showing correlation between patch intervals and player count for a specific game. """
    
    # Extract the first word from the game name and filter the data based on that
    first_word = game_name.split()[0].lower()
    game_data = df[df["Game_Name"].str.lower().str.startswith(first_word)].copy()
    
    # If no data is found for the given game name, print a message and exit the function
    if game_data.empty:
        print(f"No data found for '{game_name}'")
        return
    
    # Apply pd.to_numeric to ensure the columns are numeric
    cols = ["Avg. Players", "Gain", "prev_gap", "prev_prev_gap"]
    game_data[cols] = game_data[cols].apply(pd.to_numeric, errors="coerce")

    # Compute the correlation matrix
    correlation_matrix = game_data[cols].corr()

    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = ["Avg. # Players", "Gain #", "Patch Gap", "Prev. Patch Gap"]
    correlation_matrix.index = ["Avg. # Players", "Gain #", "Patch Gap", "Prev. Patch Gap"]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Heatmap for {game_name}")
    plt.show()




if __name__ == "__main__":
    game_data_file = "C:/Users/ashru/Downloads/game_data (1).csv"
    patch_notes_dir = "C:/Users/ashru/OneDrive/Documents/FSAI/patch_notes"

    merged_data = load_data(game_data_file, patch_notes_dir)
    print("Data loaded successfully!")

    game_name = "Apex Legends"

    plot_correlation_heatmap(merged_data)
    plot_correlation_heatmap_specific(merged_data, game_name)

