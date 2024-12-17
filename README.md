
# LLM-Improvement-with-RAG: End-to-End Chatbot Development

This project demonstrates the development of a **Retrieval-Augmented Generation (RAG)**-based chatbot using **LangChain** and **Ollama**. The chatbot can interact with PDF documents, retrieve relevant information, and provide accurate, context-aware responses.

## Features

- **PDF Chatbot**: Upload PDFs to query and retrieve answers using a conversational interface.
- **Local Vector Database**: Embeddings are stored in a local vector database (`ChromaDB`).
- **LLM Integration**: Uses the Ollama LLM (`llama3.2:1b`) for natural language processing.
- **Persistent Memory**: Conversation memory is maintained for contextual responses.

---

## Project Structure

```plaintext
LLM-Improvement-with-RAG/
│
├── app.py               # Streamlit application for chatbot
├── pdfFiles/            # Directory to store uploaded PDF files
├── vectorDB/            # Directory to store vector database
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Setup Instructions

### 1. **Install Prerequisites**

Ensure you have Python 3.8 or above installed.

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

### 2. **Install and Run Ollama**

1. **Download and install Ollama**:  
   Follow the instructions at [Ollama's website](https://ollama.ai) to install the LLM locally.

2. **Pull the LLM Model**:  
   Ensure you download the `llama3.2:1b` model:

   ```bash
   ollama pull llama3:2.1b
   ```

3. **Run Ollama Service**:  
   Start the Ollama server on `localhost:11434`:

   ```bash
   ollama serve
   ```

---

### 3. **Run the Chatbot**

Execute the Streamlit app:

```bash
streamlit run app.py
```

This will open the chatbot application in your default browser.

---

## How to Use the Application

1. **Upload PDF**: Upload a PDF file using the file uploader on the app interface.
2. **Chat with the PDF**:
   - Type a question in the chat input field.
   - The chatbot will process the PDF, retrieve relevant content, and generate a response.
3. **Conversation Memory**: The chatbot maintains the conversation history for context-aware interactions.

---
