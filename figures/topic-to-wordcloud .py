import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bertopic import BERTopic

def create_wordcloud(topic_model, topic_id, output_dir):
    # Extract the words and their importance for the given topic
    words = dict(topic_model.get_topic(topic_id))
    # Initialize a WordCloud object
    wc = WordCloud(background_color='white', width=800, height=600)
    # Generate the word cloud from frequencies
    wc.generate_from_frequencies(words)
    # Plot the word cloud
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    # Save the word cloud image to the specified directory
    plt.savefig(os.path.join(output_dir, f'topic_{topic_id}_wordcloud.png'))
    plt.close()

def main():
    # Path to the directory containing the saved BERTopic model
    model_path = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\saved_bertopic_model_v1_kmeans_guided"
    # Load the BERTopic model
    topic_model = BERTopic.load(model_path)
    
    # Directory to save the word cloud images
    output_dir = os.path.join(model_path, "wordclouds")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all topic ids, excluding the outlier topic (-1)
    topic_ids = [topic_id for topic_id in topic_model.get_topic_info().Topic if topic_id != -1]
    
    # Generate and save word clouds for each topic
    for topic_id in topic_ids:
        create_wordcloud(topic_model, topic_id, output_dir)
        print(f"Word cloud saved for topic {topic_id}")

if __name__ == "__main__":
    main()
