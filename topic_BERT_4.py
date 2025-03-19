import os
import sys
import pickle
import re
import json
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Global list for custom ignore words
CUSTOM_IGNORE_WORDS = ["http", "https", "nbsp", "apos", "quot", "STEAM_CLAN_IMAGE", "_"]

# Sample custom preprocessing function
def preprocess_doc(doc):
    # If doc is a dictionary, get the 'contents' field; otherwise, assume it's a string
    text = doc.get('contents', '') if isinstance(doc, dict) else doc
    
    # Remove game names (if provided) and generate n-grams to ignore
    game_name = doc.get('game') if isinstance(doc, dict) else None
    if game_name:
        words = game_name.split()
        ngrams = []
        for n in range(1, len(words) + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n]).lower()
                ngrams.append(ngram)
        # Update the custom ignore list
        for ngram in ngrams:
            if ngram not in CUSTOM_IGNORE_WORDS:
                CUSTOM_IGNORE_WORDS.append(ngram)
        # Remove all occurrences of these n-grams from the text
        for ngram in ngrams:
            pattern = r'\b' + re.escape(ngram) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = ' '.join(word for word in text.split() if 'http' not in word)
    # Remove common HTML tokens
    text = text.replace("&nbsp;", " ").replace("&apos;", "'").replace("&quot;", '"')
    
    tokens = word_tokenize(text)
    # Lowercase tokens that are in the ignore list and filter them out
    tokens = [word.lower() if word.lower() in CUSTOM_IGNORE_WORDS else word for word in tokens]
    tokens = [word for word in tokens if word not in CUSTOM_IGNORE_WORDS]
    # Keep only alphabetic tokens or common punctuation
    tokens = [word for word in tokens if word.isalpha() or word in ['.', ',', '!', '?']]
    
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text

BASE_DIR = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3"
MODEL_BASE_NAME = "saved_bertopic_model"
EMBEDDINGS_FILENAME = "saved_embeddings.pkl"

def get_new_version_path(base_dir, base_model_name):
    version = input("Enter a new model version name (e.g. v1, v2, etc.): ").strip()
    new_model_path = os.path.join(base_dir, f"{base_model_name}_{version}")
    if os.path.exists(new_model_path):
        print("This version already exists. Please choose a different version name.")
        return get_new_version_path(base_dir, base_model_name)
    return new_model_path

def list_model_versions(base_dir, base_model_name):
    versions = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith(base_model_name):
            versions.append(full_path)
    return versions

def run_topic_model_fitting(docs, embedding_model, model_path):
    print("Encoding docs...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    
    # Custom representation model and vectorizer with stopword removal
    representation_model = KeyBERTInspired()
    custom_umap = umap.UMAP(n_neighbors=5, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=5, gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=5)
    
    # Create a custom CountVectorizer that removes common English stopwords and can capture bi-grams
    vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
    
    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=custom_umap,
        hdbscan_model=custom_hdbscan,
        vectorizer_model=vectorizer_model,
        language="english",
        nr_topics="auto",
        calculate_probabilities=False,
        verbose=True,
    )
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    # Save the model and embeddings
    os.makedirs(model_path, exist_ok=True)
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    embeddings_path = os.path.join(model_path, EMBEDDINGS_FILENAME)
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    
    return topic_model, embeddings, topics

def main():    
    # Path to save/load the filtered data
    reduced_data_path = os.path.join(BASE_DIR, "reduced_data.pkl")

    if os.path.exists(reduced_data_path):
        print("Loading filtered data from file...")
        with open(reduced_data_path, "rb") as f:
            data, docs = pickle.load(f)
    else:
            # Load JSON data (expects a list of dictionaries)
        with open(r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets\all_patch_notes.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        data = pd.DataFrame(json_data)
        
        positive_keywords = ["update", "patch", "fix"]
        # Filter rows to keep rows where 'tags' contains 'patchnotes' OR where 'title' contains a word in list 
        data = data[data['tags'].apply(lambda x: isinstance(x, list) and 'patchnotes' in x) | data['title'].apply(lambda x: any(word in x.lower() for word in positive_keywords))]
        data = data.reset_index(drop=True)

        # Process rows and only keep those with non-empty preprocessed text
        filtered_docs = []
        filtered_records = []
        for record in data.to_dict(orient='records'):
            processed_doc = preprocess_doc(record)
            if processed_doc.strip():  # Only keep non-empty documents
                filtered_docs.append(processed_doc)
                filtered_records.append(record)

        # Update the DataFrame and docs list accordingly
        data = pd.DataFrame(filtered_records).reset_index(drop=True)
        docs = filtered_docs

        # Save the filtered data for future use
        print("Saving filtered data to file...")
        with open(reduced_data_path, "wb") as f:
            pickle.dump((data, docs), f)

    gids = data['gid'].tolist()
    dates = data['date'].tolist()
    appId = data['appid'].tolist()
    
    print("Number of patch notes:", len(data))
    print(f"Number of unique games: {len(data['appid'].unique())}")
    
    if '-y' not in sys.argv:
        user_input = input("Continue analysis with these patch notes? (yes/no): ")
        if user_input.lower().strip() not in ['yes', 'y']:
            print("Exiting analysis.")
            return

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    
    available_models = list_model_versions(BASE_DIR, MODEL_BASE_NAME)
    use_saved = False
    if available_models:
        load_choice = input("Saved topic models found. Load one? (yes/no): ").strip().lower()
        if load_choice in ['yes', 'y']:
            print("Available model versions:")
            for idx, path in enumerate(available_models):
                print(f"{idx + 1}: {os.path.basename(path)}")
            try:
                selected = int(input("Enter the number corresponding to the model version you want to load: "))
                selected_index = selected - 1
                if selected_index < 0 or selected_index >= len(available_models):
                    raise ValueError
                selected_model_path = available_models[selected_index]
                use_saved = True
            except ValueError:
                print("Invalid selection. Proceeding to fit a new model.")
    
    if use_saved:
        print("Embedding documents...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        print("Loading saved topic model...")

        topic_model = BERTopic.load(selected_model_path)
        
        print("Transforming documents...")
        batch_size = 1000
        topics = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Transforming batches"):
            batch_docs = docs[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_topics, _ = topic_model.transform(batch_docs, embeddings=batch_embeddings)
            topics.extend(batch_topics)
        else:
            print("Embeddings file not found in the selected version. Re-running fitting process.")
            new_model_path = get_new_version_path(BASE_DIR, MODEL_BASE_NAME)
            topic_model, embeddings, topics = run_topic_model_fitting(docs, embedding_model, new_model_path)
    else:
        print("No saved model loaded. Running the fitting process...")
        new_model_path = get_new_version_path(BASE_DIR, MODEL_BASE_NAME)
        topic_model, embeddings, topics = run_topic_model_fitting(docs, embedding_model, new_model_path)
    
    # # Display topic info before and after optional reductions
    # print("Before Topic Reduction:\n", topic_model.get_topic_info())
    # topic_model.reduce_topics(docs, nr_topics=5)
    # topics = topic_model.topics_

    try:
        print("Before Outlier Reduction:\n", topic_model.get_topic_info())
        topics = topic_model.reduce_outliers(docs, topics)
        print("Final Topics:\n", topic_model.get_topic_info())
    except ValueError as e:
        if str(e) == "No outliers to reduce.":
            print("No outliers were found to reduce. Skipping outlier reduction.")
        else:
            raise

    print("length of gids, topics, dates, appId:", len(gids), len(topics), len(dates), len(appId))
    results_df = pd.DataFrame({'gid': gids, 'topic': topics, 'date': dates, 'appid': appId})
    csv_path = "all_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Visualizations
    fig = topic_model.visualize_barchart(top_n_topics=len(topic_model.get_topic_info()))
    fig.show()
    fig = topic_model.visualize_topics()
    fig.show()
    topic_model.visualize_heatmap()
    topic_model.visualize_documents(docs, embeddings=embeddings)

if __name__ == "__main__":
    main()
