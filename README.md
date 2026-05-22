# Zero-Framework-RAG
A simple RAG (Retrieval-Augmented Generation) pipeline built from scratch in Python. No frameworks like LangChain, just embeddings, cosine similarity, and Gemini.
You give it a PDF. You ask a question. It finds the most relevant section and answers it.
# Customize It
Change the prompt;
  -> open main.py and edit the system_instruction block to give the model a different personality or response style.
  -> Use your own PDF — place any PDF inside the project folder and update this line in main.py:
      ```documents = extract_pdf_chunks("your_file.pdf")```
    **Note: PDF files are excluded from this repo via .gitignore. You need to bring your own.**
# Setup
Create and activate a virtual environment:
  ```python -m venv venv```
  Windows
  ```\venv\Scripts\Activate.ps1```
Install dependencies:
  ```pip install google-genai pymupdf python-dotenv```
Create a .env file in the project folder:
  ```GEMINI_API_KEY=your_key_here```
Get your free API key from [Google AI Studio](https://aistudio.google.com).
# Run
```python main.py```
# How It Works
  1. PDF is split into text chunks
  2. Each chunk is embedded using Gemini
  3. Your query is embedded the same way
  4. Cosine similarity finds the most relevant chunks
  5. Top 3 chunks are passed to Gemini with your question
