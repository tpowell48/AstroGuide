# AstroGuide: A Multimodal Astronomical Assistant

**AstroGuide** is an AI-powered assistant designed to bridge the gap between visual astronomical data and scientific textual knowledge. By integrating a fine-tuned Vision-Language Model (VLLM) with a Retrieval-Augmented Generation (RAG) system, AstroGuide can interpret complex celestial imagery and provide answers grounded in verified scientific literature.

## Features

- **Multimodal Analysis**: Upload images of nebulae, galaxies, or star clusters, and the AI will identify and describe them.
- **Grounded Explanations**: Answers are fact-checked against a database of NASA's Astronomy Picture of the Day (APOD) archives and OpenStax astronomy textbooks.
- **Interactive Agent**: Accessible via a Discord bot that maintains conversation history and context.
- **Scalable Architecture**: Built using a microservices approach with Replicate for inference and n8n for orchestration.


## System Architecture

The project consists of three core components:
1. **Vision-LLM (The Eyes)**: A **LLaVA-1.5-7b** model fine-tuned using **QLoRA** on the APOD dataset to accurately caption astronomical images.
2. **RAG System (The Brain)**: A **LlamaIndex** + **ChromaDB** pipeline utilizing **CLIP embeddings** to perform multimodal retrieval (finding text relevant to images and vice versa).
3. **AI Agent (The Orchestrator)**: An **n8n** workflow that manages user state in **Supabase** (PostgreSQL) and routes requests between the user interface and the model API.

## Repository Structure
Some items are missing from this repository due to file size limitations. <br>
![Repository Structure](RepositoryStructure)

## Setup & Installation

### Prerequisites

- Python 3.10+
- An n8n instance (Self-hosted or Cloud)
- A Replicate account (for model hosting)
- A Supabase project (for database/memory)
- A Discord Bot Token

### Model & Index Deployment

1. The inference logic is hosted on Replicate to handle the heavy lifting.
2. Use notebooks/LLaVA_Finetuning.ipynb to train the adapter weights.
3. Merge the adapters with the base model.
4. Build the RAG index using notebooks/RAG_Index_Creation.ipynb.
5. Push the Dockerized model + index to Replicate. 

### Database Setup (Supabase)

Create a table in Supabase to store chat history

### Agent Configuration (n8n)

2. Configure the Discord Trigger node and Discord Response node with your Bot Token.
3. Configure the Supabase nodes with your URL and Service Role Key.
4. Configure the HTTP Request nodes to point to your Replicate deployment endpoint.

## Usage

Once the Discord bot is online and connected to your n8n webhook:

### Text Query:
> /astroguide prompt: What is the primary factor that determines the color of a star?

### Image Analysis:
> Upload an image of a celestial body and use the slash command: /astroguide prompt: Describe this image in detail.

## Dataset
The system was trained and indexed on:
- NASA Astronomy Picture of the Day (APOD) Archive (1995-2025)
- OpenStax Astronomy 2e Textbook

## Authors
- Thomas Powell - *Lead Developer*
- Dr. Muhammad Aminul Islam - *Advisor* (University of New Haven)

## Demo
[AstroGuide AI Agent Demo](https://youtu.be/xZV4z2Z-kVs)
