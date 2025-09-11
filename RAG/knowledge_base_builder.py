import chromadb
import json
from sentence_transformers import SentenceTransformer
import os
from PIL import Image

# --- Configuration ---
OPENSTAX_JSON_PATH = 'DATA/OPENSTAX_DATA/OpenStax_Astronomy2e.json'
APOD_JSON_PATH = 'DATA/APOD_DATA/apod_data.json'
# IMPORTANT: Point this to the directory where you downloaded your APOD images
APOD_IMAGE_DIR = 'DATA/APOD_DATA/IMAGES' 
DB_PATH = 'RAG/rag_database_multimodal'
COLLECTION_NAME = 'multimodal_rag'

def add_to_collection_in_batches(collection, documents, metadatas, ids, batch_size=4000):
    """Adds documents to a ChromaDB collection in smaller batches."""
    for i in range(0, len(documents), batch_size):
        print(f"  - Adding batch {i // batch_size + 1}...")
        collection.add(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing model and vector database...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # --- Process OpenStax Data ---
    print("\n--- Processing OpenStax Textbook ---")
    try:
        with open(OPENSTAX_JSON_PATH, 'r', encoding='utf-8') as f:
            textbook_data = json.load(f)

        documents_text, metadatas_text, ids_text = [], [], []
        doc_id = 0
        print("Generating embeddings for OpenStax data...")
        for url, text_content in textbook_data.items():
            if text_content:
                chunks = text_content.split('\n')
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 10:
                        documents_text.append(chunk.strip())
                        metadatas_text.append({'source': 'openstax', 'url': url, 'type': 'text'})
                        ids_text.append(f'openstax_{doc_id}_{i}')
                doc_id += 1
        
        if documents_text:
            add_to_collection_in_batches(collection, documents_text, metadatas_text, ids_text)
            print(f"Successfully added {len(documents_text)} text chunks from OpenStax.")

    except Exception as e:
        print(f"An error occurred processing OpenStax data: {e}")

    # --- Process APOD Data ---
    print("\n--- Processing APOD Summaries and Images ---")
    try:
        with open(APOD_JSON_PATH, 'r', encoding='utf-8') as f:
            apod_data = json.load(f)

        # --- SEPARATE TEXT AND IMAGE PROCESSING ---
        text_docs, text_metadatas, text_ids = [], [], []
        image_docs, image_metadatas, image_ids = [], [], []

        for item in apod_data:
            summary = item.get('explanation')
            image_url = item.get('url')
            date = item.get('date')
            if not (summary and image_url and date): continue
            
            unique_id_base = f"{os.path.basename(image_url).split('.')[0]}_{date}"
            
            # Prepare TEXT summary
            text_docs.append(summary)
            text_metadatas.append({'source': 'apod', 'url': image_url, 'type': 'text_summary', 'title': item.get('title', ''), 'date': item.get('date', '')})
            text_ids.append(f'apod_text_{unique_id_base}')
            
            # Prepare IMAGE data
            local_image_path = os.path.join(APOD_IMAGE_DIR, f"{item.get('date')}.jpg")
            if os.path.exists(local_image_path):
                image_docs.append(local_image_path) # Placeholder for image path
                image_metadatas.append({'source': 'apod', 'url': image_url, 'type': 'image', 'title': item.get('title', ''), 'date': item.get('date', '')})
                image_ids.append(f'apod_image_{unique_id_base}')
        
        # --- Add to collection in separate steps ---
        if text_docs:
            print("\nAdding APOD text summaries...")
            add_to_collection_in_batches(collection, text_docs, text_metadatas, text_ids)
            print(f"Successfully added {len(text_docs)} APOD text items.")
        
        if image_docs:
            print("\nAdding APOD images...")
            add_to_collection_in_batches(collection, image_docs, image_metadatas, image_ids)
            print(f"Successfully added {len(image_docs)} APOD image items.")

    except Exception as e:
        print(f"An error occurred processing APOD data: {e}")

    print(f"\nKnowledge base build complete. Total items in collection: {collection.count()}")