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


#def plot_game_trends(df, game_name):
    """ Line plot showing player trends and patch gaps over time. """
    first_word = game_name.split()[0].lower()
    game_data = df[df["Game_Name"].str.lower().str.startswith(first_word)].copy()
    
    if game_data.empty:
        print(f"No data found for '{game_name}'")
        return
    
    game_data["month"] = pd.to_datetime(game_data["month"])
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_xlabel("Month", fontsize=16)
    ax1.set_ylabel("Avg. Players", color="blue", fontsize=16)
    sns.lineplot(x=game_data["month"], y=game_data["Avg. Players"], marker="o", color="blue", ax=ax1, label="Avg. Players", ci=None)
    ax1.tick_params(axis="y", labelcolor="blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Patch Interval (Days)", color="red", fontsize=16)
    
    sns.lineplot(x=game_data["month"], y=game_data["prev_gap"], color="red", marker="s", ax=ax2, label="Prev Gap", ci=None)
    sns.lineplot(x=game_data["month"], y=game_data["prev_prev_gap"], color="green", marker="^", ax=ax2, label="Prev Prev Gap", ci=None)
    
    ax2.tick_params(axis="y", labelcolor="red")
    
    plt.title(f"Patch Interval vs. Player Trends for {game_name}", fontsize=18)
    fig.tight_layout()
    plt.show()



#def plot_game_trends_change(df, game_name):
    """ Line plot showing player trends and patch gaps over time. """
    first_word = game_name.split()[0].lower()
    game_data = df[df["Game_Name"].str.lower().str.startswith(first_word)].copy()
    
    if game_data.empty:
        print(f"No data found for '{game_name}'")
        return
    
    game_data["month"] = pd.to_datetime(game_data["month"])

    # Calculate the change in players (relative to the last patch)
    game_data["change_in_players"] = game_data["Avg. Players"].diff()

    # Calculate the change in players for two patches ago
    game_data["change_in_players_2"] = game_data["Avg. Players"].diff(2)
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_xlabel("Month", fontsize=16)
    ax1.set_ylabel("Change in Players", color="blue", fontsize=16)
    sns.lineplot(x=game_data["month"], y=game_data["change_in_players"], marker="o", color="blue", ax=ax1, label="Change in Players", ci=None)
    ax1.tick_params(axis="y", labelcolor="blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Patch Interval (Days)", color="red", fontsize=16)
    
    sns.lineplot(x=game_data["month"], y=game_data["prev_gap"], color="red", marker="s", ax=ax2, label="Prev Gap", ci=None)
    sns.lineplot(x=game_data["month"], y=game_data["prev_prev_gap"], color="green", marker="^", ax=ax2, label="Prev Prev Gap", ci=None)
    
    ax2.tick_params(axis="y", labelcolor="red")
    
    plt.title(f"Patch Interval vs. Player Trends for {game_name}", fontsize=18)
    fig.tight_layout()
    plt.show()




def plot_correlation_scatter(df):
    df = df.dropna(subset=["prev_gap", "Avg. Players"])  # Remove NaNs

    # Apply log transformation to the Y-axis (player count)
    df["log_avg_players"] = np.log1p(df["Avg. Players"])  # log(1 + x) to avoid log(0)

    plt.figure(figsize=(10, 6))
    
    # Scatter plot with log-transformed Y-axis
    sns.scatterplot(x=df["prev_gap"], y=df["log_avg_players"], alpha=0.6)

    # Add a linear regression line (log scale on Y-axis)
    sns.regplot(x=df["prev_gap"], y=df["log_avg_players"], scatter=False, color="red", line_kws={"color": "blue", "lw": 2})

    # Log trendline using a logarithmic fit for the Y-axis
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["prev_gap"], df["log_avg_players"])

    # Plot the log trendline
    plt.plot(df["prev_gap"], intercept + slope * df["prev_gap"], color="green", lw=2, label="Trendline")

    plt.xlabel("Patch Interval (Days)")
    plt.ylabel("Log of Avg. Players")
    plt.title("Correlation between Patch Interval and Player Count (Log Y-Axis)")
    plt.legend()

    plt.show()



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

if __name__ == "__main__":
    game_data_file = "C:/Users/ashru/Downloads/game_data (1).csv"
    patch_notes_dir = "C:/Users/ashru/OneDrive/Documents/FSAI/patch_notes"

    merged_data = load_data(game_data_file, patch_notes_dir)
    print("Data loaded successfully!")

    game_name = "Apex Legends"

    # Generate all three visualizations
    #plot_game_trends(merged_data, game_name)
    #plot_game_trends_change(merged_data, game_name)
    plot_correlation_scatter(merged_data)
    plot_correlation_heatmap(merged_data)

