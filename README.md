# Vector RAG Chatbot Project

## Project Description

This project builds a simple Arabic RAG chatbot that answers user questions from a PDF book.

The workflow includes:

1. OCR on the book using Azure Document Intelligence
2. Chunking and text extraction
3. Creating embeddings using Sentence Transformers
4. Storing vectors in FAISS
5. Using LangChain with Ollama to answer user questions

## Project Files

* `main.ipynb`: preprocessing pipeline
* `chatbot.py`: chatbot 
* `data/books/book.pdf`: input PDF
* `data/books_proc/book1_ocr.json`: OCR output
* `data/vecs/vector_index.faiss`: vector database
* `data/vecs/documents.pkl`: stored text chunks

## How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the chatbot

```bash
python chatbot.py
```

## Technologies Used

* Python
* Azure Document Intelligence
* Sentence Transformers
* FAISS
* LangChain
* Ollama

## Notes

* The chatbot answers based only on the retrieved text.
* OCR quality affects the final answer quality.
