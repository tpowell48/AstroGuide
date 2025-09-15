import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


# --- Configuration ---
DB_PATH = "RAG/chroma_db" # Keep the DB in the RAG folder
COLLECTION_NAME = 'multimodal_knowledge_base'
QUERY_TEXT = "Show me a picture of the moon"

if __name__ == "__main__":
    # Set up the embedding model (must match the builder script)
    print("Initializing 'all-MiniLM-L6-v2' model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = None  # Disable LLM usage for querying for now

    # Connect to the existing ChromaDB collection
    print(f"Connecting to ChromaDB at '{DB_PATH}'...")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create a LlamaIndex index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create a query engine
    print("Creating query engine...")
    query_engine = index.as_query_engine(similarity_top_k=5)

    # Run the query
    print(f"\nQuery: '{QUERY_TEXT}'")
    response = query_engine.query(QUERY_TEXT)

    # Display the results
    print("\n--- Top Retrieved Results ---")
    for node in response.source_nodes:
        media_type = node.metadata.get('media_type', 'unknown')
        if media_type == 'image':
            image_path = node.metadata.get('id', 'N/A')
            print(f"Retrieved APOD Summary (Score: {node.score:.4f}):")
            print(f"  - Image: {image_path}")
            print(f"  - Summary: {node.get_content()[:200]}...")
        else:
            retrieved_text = node.get_content().replace('\n', ' ')[:200]
            print(f"Retrieved OpenStax text (Source: {media_type}, Score: {node.score:.4f}): \"{retrieved_text}...\"")