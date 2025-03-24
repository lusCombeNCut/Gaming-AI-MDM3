import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm


# -------------------------------
# Function: Perform LDA on Tags with Phrase Handling and Export to CSV
# -------------------------------
def lda_on_game_tags(csv_path, n_topics=10, filter_out_tags=None, top_n_tags=10, output_csv_path=None):
    """
    Perform LDA analysis on game tags to categorize games into topics.

    Args:
        csv_path (str): Path to merged game Twitch data CSV.
        n_topics (int): Number of topics to generate.
        filter_out_tags (list): Tags/phrases to filter out from analysis.
        top_n_tags (int): Number of top tags to display for each topic.
        output_csv_path (str): Path to save updated CSV file (optional).

    Returns:
        pd.DataFrame: Updated DataFrame with LDA topic assignments.
    """

    # Load dataset
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Default tags to filter if not provided
    if filter_out_tags is None:
        filter_out_tags = [
            "Multiplayer", "Competitive", "Action", "Strategy",
            "Military", "War", "Trading", "Realistic", "Fast-Paced", "Moddable",
            "to", "op", "co", "first", "person", "play"
        ]

    # Add additional words/phrases to filter list
    additional_words = []
    custom_stop_words = filter_out_tags + additional_words
    filter_tags_set = set(filter_out_tags)

    # Function to clean tags and keep only useful phrases
    def clean_tags(tag_str):
        tags = [tag.strip() for tag in tag_str.split(",")]
        filtered_tags = [tag for tag in tags if tag not in filter_tags_set]
        return ", ".join(filtered_tags)  # Keep comma-separated for tokenizer

    # Clean and filter tags
    print("Cleaning and filtering tags...")
    tqdm.pandas(desc="Filtering Tags")
    df['Filtered_Tags'] = df['Tags'].progress_apply(clean_tags)

    # Vectorize tags treating phrases as single tokens
    print("Vectorizing comma-separated tag phrases for LDA...")
    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(", "),  # Split by comma and space
        stop_words=custom_stop_words
    )
    tag_matrix = vectorizer.fit_transform(df['Filtered_Tags'])

    # Apply LDA
    print(f"Applying LDA to generate {n_topics} topics (this may take a moment)...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tag_matrix)

    # Assign dominant topic efficiently
    print("Assigning topics to each game...")
    topic_distributions = lda.transform(tag_matrix)
    df['LDA_Topic'] = topic_distributions.argmax(axis=1)

    # Print summary
    print(f"\n✅ LDA analysis completed. {n_topics} topics assigned.")
    print(df[['Game_Name', 'Filtered_Tags', 'LDA_Topic']].head())

    # Show top phrases per topic for interpretation
    print(f"\nTop {top_n_tags} phrases per topic (after filtering):")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[-top_n_tags:][::-1]  # Top N phrases descending
        top_features = [feature_names[i] for i in top_features_ind]
        print(f"Topic {topic_idx}: {', '.join(top_features)}")

    # Save updated dataframe to CSV if output path provided
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"\n📁 CSV file with topics saved to: {output_csv_path}")

    return df


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # File path to your merged_game_twitch_data.csv
    csv_file_path = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets\game_data.csv"
    # Output CSV path to save categorized games
    output_csv_path = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\categorized_games_with_topics.csv"
    
    # Run LDA on tags and save result to file
    updated_df = lda_on_game_tags(
        csv_path=csv_file_path,
        n_topics=7,  # Set to desired number of topics
        top_n_tags=10,  # Top phrases to display per topic
        output_csv_path=output_csv_path  # This will save the CSV
    )
