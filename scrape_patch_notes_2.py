import os
import requests
import json
import csv
from tqdm import tqdm

def load_patch_notes(directory='patch_notes', appid=None, return_all=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, directory)
    
    if not os.path.exists(directory):
        return []
    
    all_patch_notes = []
    for filename in os.listdir(directory):
        if filename.endswith('_patch_notes.json'):
            file_appid = filename.split('_')[0]
            if appid and file_appid != str(appid):
                continue
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                patch_notes = json.load(file)
                all_patch_notes.extend(patch_notes)
            if appid:
                return all_patch_notes
    return all_patch_notes if return_all else []

url_news = "http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"

def fetch_patch_notes(appid):
    params = {"appid": appid, "count": 10000, "maxlength": 30000, "format": "json"}
    response = requests.get(url_news, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'appnews' in data and 'newsitems' in data['appnews']:
            return data['appnews']['newsitems']
    return []

os.makedirs('patch_notes', exist_ok=True)

if __name__ == '__main__':
    all_combined_patch_notes = []
    
    # Aggregate unique games while tracking peak "Avg. Players" per game.
    unique_games = {}
    with open('datasets/game_data.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            game_id = row['Game_Id']
            game_name = row['Game_Name']
            try:
                avg_players = float(row['Avg. Players'])
            except ValueError:
                continue
            if game_id in unique_games:
                unique_games[game_id]['peak_avg'] = max(unique_games[game_id]['peak_avg'], avg_players)
            else:
                unique_games[game_id] = {'name': game_name, 'peak_avg': avg_players}
    
    print("number before filtering:", len(unique_games))    
    filtered_games = {appid: info for appid, info in unique_games.items() if info['peak_avg'] >= 1000}
    print("number after filtering:", len(filtered_games))

    # Iterate over games with a progress bar.
    for appid, info in tqdm(filtered_games.items(), desc="Processing games", unit="game"):
        game = info['name']
        patch_notes = fetch_patch_notes(appid)
        if patch_notes:
            english_patch_notes = []
            for note in patch_notes:
                if 'contents' in note:
                    # Optionally include game metadata for reference.
                    note['game'] = game
                    note['game_id'] = appid
                    english_patch_notes.append(note)
            # Only add games with more than 20 patch notes.
            if len(english_patch_notes) > 5:
                all_combined_patch_notes.extend(english_patch_notes)
    
    # Write all patch notes to a single JSON file.
    output_file = os.path.join('patch_notes', 'all_patch_notes.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_combined_patch_notes, f, ensure_ascii=False, indent=4)
