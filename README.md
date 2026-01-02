# 🤖 Multi-Agent Chatbot with LangGraph

An educational project teaching LLM concepts progressively: from basic API calls to LangChain to LangGraph's supervisor pattern.

![Supervisor Pattern](https://img.shields.io/badge/Pattern-Supervisor-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.26-green)
![OpenRouter](https://img.shields.io/badge/API-OpenRouter-orange)

## 🎯 What You'll Learn

| Lesson | File | Concept |
|--------|------|---------|
| 1 | `1_basic_llm_call.py` | Raw HTTP requests to LLMs |
| 2 | `2_langchain_basics.py` | LangChain chains & LCEL |
| 3 | `3_langgraph_agents.py` | LangGraph supervisor pattern |
| 4 | `4_embeddings_demo.py` | Embeddings & semantic similarity |

## 🏗️ Architecture

```
        ┌─────────────────┐
        │    Frontend     │
        │  (Chat UI)      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   FastAPI       │
        │   Backend       │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Supervisor    │
        │    Agent        │
        └────────┬────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ 🔬      │ │ 🎨      │ │ 💻      │
│Scientist│ │Creative │ │ Coder   │
└─────────┘ └─────────┘ └─────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd agentbuild

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get your key from: https://openrouter.ai/keys
```

### 3. Run the Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 4. Open the Frontend

Open `frontend/index.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 3000
```

Then visit: http://localhost:3000

## 📚 Running Individual Lessons

Each lesson can be run standalone for teaching:

```bash
cd backend

# Lesson 1: Basic LLM Call
python 1_basic_llm_call.py

# Lesson 2: LangChain Basics  
python 2_langchain_basics.py

# Lesson 3: LangGraph Agents
python 3_langgraph_agents.py

# Lesson 4: Embeddings
python 4_embeddings_demo.py
```

## 🎨 The Three Agents

| Agent | Specialty | Example Queries |
|-------|-----------|-----------------|
| 🔬 **Scientist** | Facts, research, explanations | "How does photosynthesis work?" |
| 🎨 **Creative** | Stories, poetry, imagination | "Write a poem about the moon" |
| 💻 **Coder** | Programming, technical help | "How do I write a for loop?" |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available agents |
| `/chat` | POST | Send message to multi-agent system |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing"}'
```

### Example Response

```json
{
  "agent_used": "SCIENTIST",
  "response": "🔬 As the Scientist agent, I'm happy to explain..."
}
```

## 📁 Project Structure

```
agentbuild/
├── backend/
│   ├── app.py                    # FastAPI server
│   ├── langgraph_agents.py       # Agent module for imports
│   ├── 1_basic_llm_call.py       # Lesson 1
│   ├── 2_langchain_basics.py     # Lesson 2
│   ├── 3_langgraph_agents.py     # Lesson 3
│   ├── 4_embeddings_demo.py      # Lesson 4
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── .env.example
└── README.md
```

## 🎓 Teaching Tips

### Lesson 1: Basic LLM Call
- Show the raw HTTP request/response
- Explain the message format (system, user, assistant)
- Discuss temperature and max_tokens

### Lesson 2: LangChain
- Demonstrate prompt templates
- Show chain composition with LCEL (`|` operator)
- Compare verbose vs concise syntax

### Lesson 3: LangGraph
- Draw the graph structure on a whiteboard
- Explain state management
- Show how routing decisions work

### Lesson 4: Embeddings
- Visualize vectors as points in space
- Demonstrate similarity calculations
- Connect to RAG concepts

## 🛠️ Customization

### Using Different Models

Edit the model in the Python files:

```python
# Use Claude
model="anthropic/claude-3-haiku"

# Use Llama
model="meta-llama/llama-3-8b-instruct"

# Use Mistral
model="mistralai/mistral-7b-instruct"
```

### Adding New Agents

1. Create agent function in `langgraph_agents.py`
2. Add node to the graph
3. Update supervisor prompt
4. Add routing logic

## 📝 License

MIT License - Feel free to use for teaching!

## 🙏 Credits

- [LangChain](https://python.langchain.com/)
- [LangGraph](https://python.langchain.com/docs/langgraph)
- [OpenRouter](https://openrouter.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)

