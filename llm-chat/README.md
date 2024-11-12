# LLM Chat Interface

A simple chat interface for interacting with Large Language Models (LLMs) through Ollama.

## Features

- 🚀 Real-time streaming responses
- 🎨 Web interface using Streamlit
- 🔧 Temperature control
- 📝 Chat history
- 🔄 Multiple model support

## Quick Start

1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure Ollama host in `.env`:
```
OLLAMA_HOST=http://your-ollama-host:11434
```

3. Run the app:
```bash
streamlit run main.py
```

4. Open in browser:
- http://localhost:8501

## Project Structure
```
llm_chat/
├── src/
│   ├── llm/          # Ollama client
│   ├── utils/        # Helper functions
├── config/           # Settings
├── data/            # Chat history
├── requirements.txt
└── main.py
```

## Requirements

- Python 3.11+
- Ollama running locally or remotely