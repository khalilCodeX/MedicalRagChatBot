# Medical RAG ChatBot

An AI-powered medical assistant chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate medical information based on a comprehensive database of medical cases and doctor responses.

## Features

- **RAG-based responses**: Retrieves relevant medical cases before generating answers
- **Vector search**: Uses Qdrant for efficient similarity search
- **OpenAI integration**: Powered by GPT models for intelligent responses
- **Document chunking**: Efficiently processes large medical datasets
- **Token tracking**: Monitors API usage and costs
- **Interactive chat**: Command-line interface for easy interaction

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │
│  (OpenAI)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │
│  (Qdrant)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similar Docs   │
│  Retrieval      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Context    │
│  Formation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPT Response   │
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Response  │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- OpenAI API key
- Qdrant account (cloud or local instance)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/khalilCodeX/MedicalRagChatBot.git
   cd MedicalRagChatBot
   ```

2. **Download the dataset**
   
   Download the medical chatbot dataset and save it as `ai-medical-chatbot.csv` in the project root.
   
   Dataset sources:
   - [Kaggle Medical Datasets](https://www.kaggle.com/datasets)
   - Or your own medical Q&A dataset in CSV format

   Required CSV columns:
   - `Description`: Medical case description
   - `Patient`: Patient query/question
   - `Doctor`: Doctor's response

3. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   QDRANT_URL=your_qdrant_url_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_COLLECTION_NAME=medical_chatbot
   ```

## Usage

### First-time setup (Embed documents)

Before using the chatbot for the first time, you need to embed the medical documents into the vector database:

```python
# In chat.py, uncomment these lines for first run:
doc_chunks = bot.tokenize_chunkify_documents()
bot.embed_documents(doc_chunks)
```

Run:
```bash
python chat.py
```

This will:
- Load the CSV dataset
- Chunk documents into manageable pieces
- Create embeddings using OpenAI
- Store vectors in Qdrant

### Regular usage

After initial setup, comment out the embedding lines and use:

```bash
python chat.py
```

Then interact with the chatbot:
```
Enter your medical query (or type 'q' to quit): What are the symptoms of diabetes?
AI Response: [AI-generated response based on retrieved medical cases]

Enter your medical query (or type 'q' to quit): q
Exiting the chat. Goodbye!
```

## Project Structure

```
MedicalRagChatBot/
├── chat.py              # Main chat interface
├── dataloader.py        # Data loading and document processing
├── genai_llm.py         # LLM chain and prompt management
├── vectordb.py          # Qdrant vector database operations
├── management.py        # OpenAI and Qdrant client setup
├── token_calc.py        # Token counting and cost estimation
├── .env                 # Environment variables (not in git)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Components

### `dataloader.py`
- Loads medical dataset from CSV
- Creates LangChain documents with metadata
- Chunks documents for efficient processing

### `vectordb.py`
- Manages Qdrant vector store
- Embeds documents using OpenAI embeddings
- Performs similarity search for relevant documents

### `genai_llm.py`
- Creates LLM chains with context
- Formats retrieved documents with metadata
- Generates system and user prompts

### `management.py`
- Handles OpenAI client initialization
- Manages Qdrant client connection

### `token_calc.py`
- Calculates token counts for text
- Estimates API costs

## Configuration

### Document Chunking
Default settings in `dataloader.py`:
```python
chunk_size=1000  # Characters per chunk
overlap=200      # Overlap between chunks
```

### Vector Search
Default retrieval in `vectordb.py`:
```python
k=5  # Number of similar documents to retrieve
```

### Embeddings
Default model in `vectordb.py`:
```python
model="text-embedding-3-small"
```

## Cost Estimation

The chatbot tracks token usage and estimates costs:
- Input tokens: $0.05 per 1M tokens
- Output tokens: $0.40 per 1M tokens (estimated as 2x input)

## Disclaimer

**IMPORTANT**: This chatbot is for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## License

MIT License - feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please open an issue on GitHub.

---

Built with LangChain, OpenAI, and Qdrant
