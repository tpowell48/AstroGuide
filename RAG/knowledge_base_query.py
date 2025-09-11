import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

# --- Configuration ---
DB_PATH = 'RAG/rag_database_multimodal'
COLLECTION_NAME = 'multimodal_rag'
APOD_IMAGE_DIR = 'DATA/APOD_DATA/IMAGES' # Path to downloaded APOD images

if __name__ == "__main__":
    # Initialize the CLIP model and connect to the database
    print("Initializing CLIP model and connecting to vector database...")
    model = SentenceTransformer('clip-ViT-B-32')
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Define the user's query
    query_text = "What is a nebula?"
    print(f"\nUser Query: '{query_text}'")

    # Create an embedding for the query
    query_embedding = model.encode(query_text)

    # Query the collection to find the most similar items
    print("\nRetrieving relevant documents and images from the knowledge base...")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5 # Retrieve the top 5 most relevant results
    )

    # 5. Process and display the results
    print("\n--- Top 5 Retrieved Results ---")
    if not results['ids'][0]:
        print("No results found.")
    else:
        for i, doc_id in enumerate(results['ids'][0]):
            doc_type = results['metadatas'][0][i]['type']
            source = results['metadatas'][0][i]['source']
            
            print(f"\n{i+1}. Type: {doc_type} (Source: {source})")
            
            if doc_type == 'image':
                # For images, show the title and the local file path
                title = results['metadatas'][0][i]['title']
                date = results['metadatas'][0][i]['date']
                image_path = os.path.join(APOD_IMAGE_DIR, f"{date}.jpg")
                print(f"   Title: {title}")
                print(f"   Image Path: {image_path}")
                # Image.open(image_path).show()
            else:
                # For text, show the content
                content = results['documents'][0][i]
                print(f"   Content: {content}")