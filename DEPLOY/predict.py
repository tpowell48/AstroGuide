import os
import json
import base64
import requests
import chromadb
import time
import subprocess
import concurrent.futures
from cog import BasePredictor, Input, Path
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.clip import ClipEmbedding

# --- Configuration ---
LLAVA_MODEL_NAME = "tpowe2/astroguide"  # HuggingFace model name
DB_PATH = "multimodal_db"
INDEX_PERSIST_DIR = "index"
APOD_JSON_PATH = "apod_data.json"
#VLLM server will run inside the container
LLAVA_API_BASE = "http://127.0.0.1:8000/v1"

# --- VLLM QUERY FUNCTION ---
def query_llava_server(prompt_text: str, image_path: str = None, history_messages: list = None) -> str:
    """Sends a prompt (and optionally an image) to the VLLM server."""
    print(f"\n--- Sending request to VLLM server at {LLAVA_API_BASE} ---")
    headers = {"Content-Type": "application/json"}
    content_list = [{"type": "text", "text": prompt_text}]

    all_messages_for_vllm = history_messages if history_messages is not None else []

    if image_path:
        print(f"--- Encoding image from: {image_path} ---")
        try:
            with open(image_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        except Exception as e:
            print(f"Error processing image file: {e}")
            return f"Error: Could not process image file at {image_path}."

    
    all_messages_for_vllm.append({"role": "user", "content": content_list})
    
    payload = {
        "model": LLAVA_MODEL_NAME,
        "messages": all_messages_for_vllm,
        "max_tokens": 500,
        "temperature": 0.7,
    }

    try:
        response = requests.post(f"{LLAVA_API_BASE}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLaVA server: {e}")
        return f"Error calling LLaVA server: {e}"


# --- RAG LOADING FUNCTION ---
def load_rag_components(db_path, persist_dir):
    """Loads and returns the RAG index and the CLIP embedding model."""
    print("--- Loading RAG Components from local path ---")
    db = chromadb.PersistentClient(path=db_path)
    text_collection = db.get_collection("text_collection")
    image_collection = db.get_collection("image_collection")
    text_store = ChromaVectorStore(chroma_collection=text_collection)
    image_store = ChromaVectorStore(chroma_collection=image_collection)
    clip_embed = ClipEmbedding() 

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store, persist_dir=persist_dir
    )
    index = load_index_from_storage(
        storage_context, 
        embed_model=clip_embed, 
        image_embed_model=clip_embed
    )
    print("--- RAG Components loaded successfully ---")
    return index, clip_embed


# --- AGENT ORCHESTRATOR ---
def query_astroguide_agent(
    prompt_text: str,
    image_path: str = None,
    index=None,
    clip_embed=None,
    apod_json_path=APOD_JSON_PATH,
    history_list: list = None
) -> tuple[str, str | None]: # Return type changed to tuple
    """
    Orchestrates the full RAG-VLLM query using manual vector store queries.
    Returns a tuple: (text_response, retrieved_image_path_or_None)
    """
    if index is None or clip_embed is None:
        return "Error: RAG index or embedder not provided.", None

    augmented_prompt = prompt_text
    context_str = "No context retrieved."
    retrieved_file_path_for_output = None # Variable to store the path we want to return

    if image_path:
        # --- MULTIMODAL (IMAGE RAG) PATH ---
        print(f"\n--- Executing Multimodal RAG Query ---")
        try:
            query_embedding = clip_embed.get_image_embedding(img_file_path=image_path)
            query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
            query_result = index.image_vector_store.query(query)

            if query_result.nodes:
                # Store this path to return it later
                original_retrieved_path = query_result.nodes[0].metadata['file_path']
                print(f"--- RAG retrieved image path: {original_retrieved_path} ---")

                image_name = os.path.basename(os.path.normpath(original_retrieved_path))
                target_date = image_name.removesuffix(".jpg")

                # Construct the path *inside the container*
                # Assumes your cog.yaml includes "DATA/IMAGES/"
                container_image_path = os.path.join("IMAGES", image_name)

                # Use this container path for the os.path.exists check and for returning
                retrieved_file_path_for_output = container_image_path

                with open(apod_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                image_explanation = "Explanation not found."
                for item in data:
                    if item.get('date') == target_date:
                        image_explanation = item.get('explanation')
                        break
                context_str = image_explanation

            augmented_prompt = (
                f"Context information is below.\n---------------------\n"
                f"{context_str}\n---------------------\n"
                f"Given the context information and prior knowledge, answer the query.\n"
                f"Query:{prompt_text} .\nAnswer: "
            )
        except Exception as e:
            print(f"Error during image RAG: {e}")
            augmented_prompt = prompt_text # Fallback

    else:
        # --- TEXT-ONLY RAG PATH ---
        print(f"\n--- Executing Text-only RAG Query ---")
        try:
            query_embedding = clip_embed.get_text_embedding(text=prompt_text)
            query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)
            query_result = index.vector_store.query(query)

            if query_result.nodes:
                retrieved_text = [node.text for node in query_result.nodes]
                context_str = ' '.join(retrieved_text)

            augmented_prompt = (
                f"Context information is below.\n---------------------\n"
                f"{context_str}\n---------------------\n"
                f"Given the context information and prior knowledge, answer the query.\n"
                f"Query:{prompt_text} .\nAnswer: "
            )
        except Exception as e:
            print(f"Error during text RAG: {e}")
            augmented_prompt = prompt_text # Fallback

    # --- FINAL CALL TO VLLM ---
    print(f"--- Sending to LLaVA with final prompt:\n{augmented_prompt[:200]}...")
    final_text_response = query_llava_server(
        prompt_text=augmented_prompt,
        image_path=image_path, # Send the *user's* image to LLaVA if provided
        history_messages=history_list
    )

    # Return both the text response and the path of the retrieved image (or None)
    return final_text_response, retrieved_file_path_for_output


# --- MAIN COG PREDICTOR CLASS ---

class Predictor(BasePredictor):

    def _start_vllm_server(self):
        """Helper method to start the VLLM server and wait for readiness."""
        print("--- Starting VLLM server in background... ---")
        
        vllm_command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", LLAVA_MODEL_NAME,
            "--trust-remote-code",
            "--port", "8000",
            "--host", "127.0.0.1",
            "--enforce-eager",
            "--gpu-memory-utilization", "0.7"
        ]
        
        # Start the server as a background process
        vllm_server_process = subprocess.Popen(vllm_command)
        
        # --- Wait for VLLM server to be ready ---
        start_time = time.time()
        timeout = 300 # Keep a dedicated timeout for VLLM load
        server_ready = False
        
        print("--- Waiting for VLLM server to load model... ---")
        while time.time() - start_time < timeout:
            try:
                # Poll the health endpoint aggressively
                response = requests.get(f"{LLAVA_API_BASE.replace('/v1', '')}/health")
                if response.status_code == 200:
                    server_ready = True
                    print("--- VLLM Server is ready! ---")
                    break
            except requests.ConnectionError:
                pass
            time.sleep(5)

        if not server_ready:
            print("--- VLLM server failed to start! ---")
            vllm_server_process.terminate()
            raise RuntimeError("VLLM server failed to start within the 300s timeout.")
        
        return vllm_server_process

    def setup(self):

        print("--- Starting concurrent setup tasks ---")
        
        # Create a thread pool with max_workers=2 for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            
            # Task 1 (Heavy I/O/GPU): Start the VLLM server
            vllm_future = executor.submit(self._start_vllm_server)

            # Task 2 (Heavy Disk I/O/CPU RAM): Load RAG components
            rag_future = executor.submit(load_rag_components, DB_PATH, INDEX_PERSIST_DIR)

            # --- Wait for results ---
            
            # Retrieve the VLLM server process handle
            self.vllm_server_process = vllm_future.result() 

            # Retrieve the RAG index and embedder objects
            self.index, self.clip_embed = rag_future.result()
            
        print("--- All components loaded concurrently and successfully! ---")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model"),
        image: Path = Input(description="Input image (optional)", default=None),
        chat_history: str = Input(description="Chat history in JSON format", default=None)
    ) -> dict: # Changed return type to dict
        """
        Run prediction, returning text response and the retrieved RAG image (if applicable).
        """
        image_path_str = str(image) if image else None

        history_list = json.loads(chat_history) if chat_history else []

        # Call the agent, which now returns a tuple
        text_response, retrieved_image_path = query_astroguide_agent(
            prompt_text=prompt,
            image_path=image_path_str,
            index=self.index,
            clip_embed=self.clip_embed,
            history_list=history_list
        )

        # Prepare the output dictionary
        output = {"text_output": text_response}

        # If the agent returned a retrieved image path...
        if retrieved_image_path:
            # Check if the file actually exists inside the container
            # The path from ChromaDB metadata might be absolute or relative
            # We need to make sure it's accessible from the current working dir
            if os.path.exists(retrieved_image_path):
                 print(f"--- Attaching retrieved image: {retrieved_image_path} ---")
                 # Convert the string path to a cog.Path object for output
                 output["retrieved_image"] = Path(retrieved_image_path)
            else:
                 print(f"--- WARNING: Retrieved image path not found: {retrieved_image_path} ---")
                 # Optionally add a note to the output
                 output["retrieved_image_status"] = f"File not found at path: {retrieved_image_path}"


        return output