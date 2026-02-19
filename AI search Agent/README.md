# AI Web Search Agent

An AI-powered web search agent built with Python, LangChain, OpenAI, and Tavily. Search the web using natural language and get synthesized answers with cited sources.

## Features

- **Web search**: Uses Tavily to search the web for up-to-date information
- **AI synthesis**: OpenAI (GPT-4o-mini) synthesizes results into clear, concise answers
- **CLI**: Interactive and single-shot modes for terminal use
- **Web UI**: Streamlit interface for browser-based searches

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy the example environment file and add your API keys:

```bash
copy .env.example .env   # Windows
# or: cp .env.example .env   # macOS/Linux
```

Edit `.env` and set:

- **OPENAI_API_KEY**: Your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
- **TAVILY_API_KEY**: Your Tavily API key from [app.tavily.com](https://app.tavily.com/sign-in) (free tier: 1,000 searches/month)

## Usage

### CLI

**Interactive mode** (prompt for queries in a loop):

```bash
python cli.py
```

**Single-shot mode** (run one query and exit):

```bash
python cli.py --query "What is the capital of France?"
# or
python cli.py -q "Latest news about AI"
```

### Web UI

```bash
streamlit run app.py
```

Then open the URL shown in your browser (typically http://localhost:8501).

## Project structure

```
AI search Agent/
├── .env.example       # Template for API keys
├── .gitignore
├── README.md
├── requirements.txt
├── agent.py           # Core agent logic (LangChain + OpenAI + Tavily)
├── cli.py             # CLI entry point
└── app.py             # Streamlit web app
```

## Tech stack

- **LangChain**: Agent framework
- **OpenAI** (GPT-4o-mini): LLM for synthesis
- **Tavily**: Web search API
- **Streamlit**: Web UI
- **python-dotenv**: Environment variable loading
