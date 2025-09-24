import json
import pandas as pd
import chromadb
import os
from sentence_transformers import SentenceTransformer

# --- Configuration ---
OPENSTAX_JSON_PATH = 'DATA/OPENSTAX_DATA/OpenStax_Astronomy2e.json'
APOD_JSON_PATH = 'DATA/APOD_DATA/apod_data.json'
APOD_IMAGE_DIR = 'DATA/APOD_DATA/IMAGES' # The folder where your APOD images are saved

def text_to_embedding(text, model):
    """Generates a vector embedding for a given text."""
    text = text.replace("\n", " ")
    return model.encode(text)

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize a text-embedding model
    print("Initializing embedding model...")
    # Using a text-specific model is more efficient for this task
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create an empty DataFrame and a list to hold all new rows
    df = pd.DataFrame(columns=['id', 'media_type', 'text', 'embeddings'])
    all_new_rows = []

    # --- Process OpenStax Textbook Data ---
    print(f"\n--- Loading and processing '{OPENSTAX_JSON_PATH}' ---")
    try:
        with open(OPENSTAX_JSON_PATH, 'r', encoding='utf-8') as f:
            textbook_data = json.load(f)

        for module_id, text_content in textbook_data.items():

            if text_content and isinstance(text_content, str):
                embedding = text_to_embedding(text_content, embedding_model)
                new_row = {
                    'id': module_id,
                    'media_type': 'text',
                    'text': text_content,
                    'embeddings': embedding
                }
                all_new_rows.append(new_row)
    except FileNotFoundError:
        print(f"Warning: File not found at '{OPENSTAX_JSON_PATH}'. Skipping.")

    # --- Process APOD Explanation Data ---
    print(f"\n--- Loading and processing '{APOD_JSON_PATH}' ---")
    try:
        with open(APOD_JSON_PATH, 'r', encoding='utf-8') as f:
            apod_data = json.load(f)

        for item in apod_data:
            date = item.get('date')
            summary = item.get('explanation')
            
            if date and summary:
                image_path = os.path.join(APOD_IMAGE_DIR, f"{date}.jpg")
                
                # We only add an entry if the corresponding image exists locally
                if os.path.exists(image_path):
                    # Generate an embedding for the TEXT EXPLANATION
                    embedding = text_to_embedding(summary, embedding_model)
                    
                    new_row = {
                        'id': image_path, # The ID is the path to the image
                        'media_type': 'image', # Signifies this text describes an image
                        'text': summary,
                        'embeddings': embedding
                    }
                    all_new_rows.append(new_row)
    except FileNotFoundError:
        print(f"Warning: File not found at '{APOD_JSON_PATH}'. Skipping.")

    # Concatenate all rows into the DataFrame
    if all_new_rows:
        df = pd.concat([df, pd.DataFrame(all_new_rows)], ignore_index=True)
        
    # Set up the ChromaDB client and collection
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="RAG/chroma_db")
    collection = client.get_or_create_collection(name="multimodal_knowledge_base")

    # Prepare the data for ChromaDB from your DataFrame
    # ChromaDB needs lists of ids, embeddings, metadatas, and documents
    ids = [f"item_{i}" for i in range(len(df))]
    embeddings = df['embeddings'].tolist()
    metadatas = df[['id', 'media_type']].to_dict('records')
    documents = df['text'].tolist()

    # Add the data to the collection
    print(f"Adding {len(df)} items to the collection...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

    print(f"Success! The collection now contains {collection.count()} items.")