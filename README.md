# CBAM Chatbot with Retrieval Augmented Generation (RAG)

## Overview

This project creates an intelligent system that enhances information retrieval from CBAM-related documents through natural language queries. It leverages the Retrieval Augmented Generation (RAG) methodology introduced by Meta AI researchers.

## Retrieval Augmented Generation (RAG)

### Introduction

RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an information retrieval component with a text generator model to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for fine-tuning and modification of internal knowledge without extensive retraining.

### Workflow

1. **Input**: RAG takes multiple PDFs as input.
2. **VectoreStore**: The PDFs are converted to vectorstore using FAISS and all-MiniLM-L6-v2 Embeddings model from Hugging Face.
3. **Memory**: Conversation buffer memory is used to maintain a track of previous conversation which are fed to the LLM model along with the user query.
4. **Text Generation with GPT-3.5 Turbo**: The embedded input is fed to the GPT-3.5 Turbo model from the OpenAI API, which produces the final output.
5. **User Interface**: Streamlit is used to create the interface for the application.

### Workflow
You must use Python 3.9.

## Getting Started

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/nina-tan/CBAM-Chatbot.git
   ```

2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory containing `OPENAI_API_KEY = [insert key here]`.

4. Run the application.
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8000` to access the user interface.

Based on: https://github.com/ArmaanSeth/ChatPDF