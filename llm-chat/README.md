# LLM Chat Interface

A streamlined chat interface for interacting with Large Language Models (LLMs) through Ollama, featuring Redis for chat history persistence.

## Features

- 🚀 Real-time streaming responses
- 🎨 Clean web interface with Streamlit
- 🔧 Adjustable temperature control
- 📝 Chat history with Redis persistence
- 🔄 Multiple model support
- 🌐 Works with local or remote Ollama instances

## Prerequisites

- Python 3.11+
- Ollama running (local or remote)
- Docker (for Redis)

## Quick Start

1. Clone and setup:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Start Redis using Docker:
```bash
docker-compose up -d
```

3. Configure environment variables in `.env`:
```env
OLLAMA_HOST=http://your-ollama-host:11434
REDIS_HOST=localhost
REDIS_PORT=6379
```

4. Run the application:
```bash
streamlit run main.py
```

5. Open in browser:
- http://localhost:8501

## Features Details

### Chat Interface
- Stream responses in real-time
- Adjust temperature for response randomness
- View and manage chat history
- Create new chat sessions
- Delete old sessions

### History Management
- Persistent storage with Redis
- Multiple chat sessions
- Session metadata tracking
- Easy session switching

## Environment Setup

### Required Environment Variables
```env
OLLAMA_HOST=http://your-ollama-host:11434   # Ollama API endpoint
REDIS_HOST=localhost                        # Redis host
REDIS_PORT=6379                             # Redis port
```

### Docker Compose
```yaml
version: '3.8'
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
```

## Development

### Requirements
```bash
streamlit>=1.31.0
requests>=2.31.0
pyyaml>=6.0.1
python-dotenv>=1.0.0
redis>=5.0.1
```

### Running Tests
```bash
# Coming soon
```

## Troubleshooting

### Common Issues

1. Ollama Connection
```bash
# Test Ollama connection
curl http://your-ollama-host:11434/api/tags
```

2. Redis Connection
```bash
# Check Redis container
docker ps | grep redis

# Test Redis connection
redis-cli ping
```

3. Streamlit
```bash
# Clear Streamlit cache
streamlit cache clear
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License