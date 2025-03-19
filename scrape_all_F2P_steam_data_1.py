import os
import time
import requests
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm  # progress bar

def get_free_to_play_games():
    url = 'https://store.steampowered.com/search/?maxprice=free&category1=998&ndl=1'
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(2)

    games = set()
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        game_elements = soup.find_all('a', class_='search_result_row')

        for game in game_elements:
            title = game.find('span', class_='title').text.strip()
            games.add(title)

        print(f"Number of games found: {len(games)}")

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.quit()
    return list(games)

def save_games_list(games, filename):
    with open(filename, 'wb') as file:
        pickle.dump(games, file)
    print(f"Saved list of {len(games)} games to {filename}")

def load_games_list(filename):
    with open(filename, 'rb') as file:
        games = pickle.load(file)
    print(f"Loaded list of {len(games)} games from {filename}")
    return games

def get_steam_app_id(game_name):
    search_url = f"https://steamcommunity.com/actions/SearchApps/{game_name}"
    response = requests.get(search_url)
    if response.status_code == 200 and response.json():
        return response.json()[0]["appid"]
    else:
        print(f"Could not find App ID for '{game_name}'")
        return None

def get_existing_game_ids(filename):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            return set(df['Game_Id'].astype(str)) if 'Game_Id' in df.columns else set()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return set()
    return set()

def get_last_saved_index(filename):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            unique_game_ids = df['Game_Id'].unique()
            return len(unique_game_ids)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return 0
    return 0

def get_steam_tags(game_id):
    url = f"https://store.steampowered.com/app/{game_id}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        game_name_tag = soup.find("div", class_="apphub_AppName")
        game_name = game_name_tag.text.strip() if game_name_tag else f"Unknown ({game_id})"
        tags = [tag.text.strip() for tag in soup.find_all("a", class_="app_tag")]
        return game_name, ", ".join(tags)
    else:
        print(f"Failed to fetch Steam tags for {game_id}")
        return f"Unknown ({game_id})", "N/A"

def get_steamcharts_data(game_id):
    url = f"https://steamcharts.com/app/{game_id}#All"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "common-table"})
        if table:
            rows = table.find_all("tr")
            headers = [th.text.strip() for th in rows[0].find_all("th")]
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) == len(headers):
                    data.append([col.text.strip() for col in cols])
            return pd.DataFrame(data, columns=headers)
    return None

def save_game_data(game_name, existing_game_ids):
    game_id = get_steam_app_id(game_name)
    if not game_id or str(game_id) in existing_game_ids:
        # print(f"Skipping '{game_name}', data already exists.")
        return
    
    game_name, tags = get_steam_tags(game_id)
    steamcharts_data = get_steamcharts_data(game_id)

    if steamcharts_data is not None:
        steamcharts_data.insert(0, "Game_Id", game_id)
        steamcharts_data.insert(1, "Game_Name", game_name)
        steamcharts_data.insert(2, "Tags", tags)
        filename = "datasets/game_data.csv"
        file_exists = os.path.exists(filename)
        steamcharts_data.to_csv(filename, mode='a', header=not file_exists, index=False)
        # print(f"Data for '{game_name}' saved successfully to {filename}")

if __name__ == "__main__":
    games_list_file = 'datasets/free_to_play_games.pkl'

    if os.path.exists(games_list_file):
        # Tell use how many unique game are in the exisiting list and ask if they want to load it
        free_games = load_games_list(games_list_file)
        choice = input(f"'{games_list_file}' already exists. With length {len(free_games)}. Do you want to load the list from the file? (yes/no): ").strip().lower()
        if choice == 'yes':
            free_games = load_games_list(games_list_file)
        else:
            free_games = get_free_to_play_games()
            save_games_list(free_games, games_list_file)
    else:
        free_games = get_free_to_play_games()
        save_games_list(free_games, games_list_file)

    existing_game_ids = get_existing_game_ids()
    print(f"Found {len(free_games)} free-to-play games.")

    # Determine the number of games already saved and resume from there
    last_saved_index = get_last_saved_index()
    print(f"Resuming from game index {last_saved_index}.")

    max_games_to_scrape = len(free_games) - last_saved_index
    num_games = int(input(f"How many games would you like to scrape data for? (Max: {max_games_to_scrape}) "))

    selected_games = free_games[last_saved_index:last_saved_index + num_games]
    
    # Process games with a progress bar showing estimated remaining time
    for game in tqdm(selected_games, desc="Processing games", unit="game"):
        save_game_data(game, existing_game_ids)
