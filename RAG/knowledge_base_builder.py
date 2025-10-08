import chromadb
import os
import json
from PIL import Image, UnidentifiedImageError

from llama_index.core import Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.core.schema import ImageDocument
from llama_index.core.indices import MultiModalVectorStoreIndex


# --- Configuration ---
OPENSTAX_JSON_PATH = 'DATA/OPENSTAX_DATA/OpenStax_Astronomy2e.json'
APOD_JSON_PATH = 'DATA/APOD_DATA/apod_data.json'
APOD_IMAGE_DIR = 'DATA/APOD_DATA/IMAGES'
CHROMA_DB_PATH = 'RAG/multimodal_db'

if __name__ == "__main__":
    # --- Set up the CLIP Multimodal Embedding Model ---
    print("Initializing CLIP embedding model...")
    Settings.embed_model = ClipEmbedding()

    # Use a text splitter with a small chunk size required by CLIP
    text_parser = SentenceSplitter(
        chunk_size=53,
        chunk_overlap=5,
    )

    # --- Load and Prepare All Documents and Nodes ---
    print("Loading all text and image documents...")

    text_documents_to_parse = []
    image_documents = []

    # Load APOD data
    try:
        with open(APOD_JSON_PATH, 'r', encoding='utf-8') as f:
            apod_data = json.load(f)
        for item in apod_data:
            summary, date = item.get('explanation'), item.get('date')
            image_path = os.path.join(APOD_IMAGE_DIR, f"{date}.jpg")

            # Add the text summary to the list of docs to be parsed
            if summary and summary.strip():
                text_documents_to_parse.append(Document(text=summary, metadata={'source': 'apod_summary', 'date': date}))

            # Add the image document if it exists and is not corrupt
            if os.path.exists(image_path):
                try:
                    Image.open(image_path) # A quick check for corruption
                    image_documents.append(ImageDocument(image_path=image_path, metadata={'source': 'apod_image', 'file_path': image_path}))
                except (UnidentifiedImageError, IOError):
                    print(f"Warning: Corrupted image skipped: {image_path}")
    except FileNotFoundError:
        print(f"Warning: APOD JSON file not found at '{APOD_JSON_PATH}'.")

    # Load OpenStax text
    try:
        with open(OPENSTAX_JSON_PATH, 'r', encoding='utf-8') as f:
            textbook_data = json.load(f)
        # Filter out any empty text entries before joining
        full_text = "\n\n".join(value for value in textbook_data.values() if value and value.strip())
        if full_text:
            text_documents_to_parse.append(Document(text=full_text, metadata={'source': 'openstax'}))
    except FileNotFoundError:
        print(f"Warning: OpenStax file not found at '{OPENSTAX_JSON_PATH}'.")

    # --- Parse Text and Combine with Images ---
    print(f"Parsing {len(text_documents_to_parse)} text documents into smaller nodes...")
    raw_text_nodes = text_parser.get_nodes_from_documents(text_documents_to_parse)
    # Filter out any nodes that might be empty after parsing
    text_nodes = [node for node in raw_text_nodes if node.get_content().strip()]

    # Combine the parsed text nodes with the image documents
    all_nodes = text_nodes + image_documents
    print(f"Prepared a total of {len(all_nodes)} nodes for indexing.")

    # --- Set up ChromaDB and Build the Index ---
    if not all_nodes:
        print("Error: No valid documents or nodes were prepared.")
    else:
        print(f"Setting up ChromaDB at '{CHROMA_DB_PATH}'...")
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        chroma_collection = db.get_or_create_collection("multimodal_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Insert nodes in index
    print("Creating an index and preparing to insert nodes...")
    index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            show_progress=True
    )
        
    print("Index building complete!")
    print(f"Verification: Database now contains {chroma_collection.count()} items.")
    
    # Verification for images
    image_items = chroma_collection.get(where={"source": "apod_image"})
    print(f"Verification: Found {len(image_items['ids'])} image embeddings in the database.")