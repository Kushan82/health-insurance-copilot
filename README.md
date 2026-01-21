# Create a minimal README.md
cat > README.md << 'EOF'
# Health Insurance Copilot

AI-powered health insurance advisory chatbot.

## Status

🚧 Under Development

## Tech Stack

- Python 3.11+
- FastAPI
- LangChain + Ollama
- ChromaDB
- LangSmith

## Setup

```bash
# Create virtual environment
uv venv --python python3

# Activate
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e ".[dev]"
