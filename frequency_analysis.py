import pandas as pd
import json
import os

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

def some_graphing():
    pass

if __name__ == "__main__":
    game_data = pd.read_csv("C:/Uni/MDM3/gaming/datasets/game_data.csv", encoding="ISO-8859-1")

    # Rename 'Month' to 'month' and format as YYYY-MM
    game_data.rename(columns={"Month": "month", "Game_Id": "appid"}, inplace=True)
    game_data["month"] = pd.to_datetime(game_data["month"], format="%B %Y").dt.strftime("%Y-%m")

    patch_notes_dir = "C:/Uni/MDM3/gaming/patch_notes"
    patch_counts = process_patch_notes(patch_notes_dir)

    merge_data(game_data, patch_counts)
