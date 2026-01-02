"""
🎓 LESSON 3: LangGraph Supervisor Pattern
==========================================
LangGraph enables building stateful, multi-agent systems.

The SUPERVISOR pattern:
- One "boss" agent that routes tasks
- Multiple specialized "worker" agents
- State flows through a graph structure

Our agents:
🔬 Scientist - Factual/research questions
🎨 Creative - Stories, poetry, creative writing  
💻 Coder - Programming and code questions
"""

import os
from typing import Annotated, Literal, TypedDict, Sequence
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

load_dotenv()

# Initialize LLM with OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost:3000"}
)


# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """
    The state that flows through our graph.
    
    - messages: Conversation history
    - next_agent: Which agent should handle next
    - final_response: The completed response
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    final_response: str


# ============================================
# SUPERVISOR AGENT
# ============================================

def create_supervisor():
    """
    The Supervisor decides which specialist should handle the query.
    
    This is the "brain" of the multi-agent system!
    """
    
    system_prompt = """You are a supervisor managing a team of specialists.
    
Your team:
- SCIENTIST: Handles factual questions, science, research, explanations of how things work
- CREATIVE: Handles stories, poetry, creative writing, jokes, imaginative content
- CODER: Handles programming questions, code writing, debugging, technical implementations

Analyze the user's message and decide which specialist should handle it.
Respond with ONLY one word: SCIENTIST, CREATIVE, or CODER"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain


def supervisor_node(state: AgentState) -> dict:
    """Supervisor node - routes to appropriate agent"""
    supervisor = create_supervisor()
    
    # Get the last human message
    last_message = state["messages"][-1].content
    
    # Decide which agent
    decision = supervisor.invoke({"input": last_message}).strip().upper()
    
    # Validate decision
    valid_agents = ["SCIENTIST", "CREATIVE", "CODER"]
    if decision not in valid_agents:
        decision = "SCIENTIST"  # Default fallback
    
    print(f"🎯 Supervisor Decision: Route to {decision}")
    
    return {"next_agent": decision}


# ============================================
# SPECIALIST AGENTS
# ============================================

def create_scientist_agent():
    """🔬 The Scientist Agent - Facts, research, explanations"""
    
    system_prompt = """You are a brilliant scientist and educator.
You explain complex topics clearly and accurately.
You love sharing fascinating facts and breaking down how things work.
Always cite that you're the Scientist agent at the start.
Be informative but concise (2-3 paragraphs max)."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()


def create_creative_agent():
    """🎨 The Creative Agent - Stories, poetry, imagination"""
    
    system_prompt = """You are a creative writing genius and storyteller.
You craft beautiful prose, poetry, and imaginative content.
Your writing is vivid, engaging, and emotionally resonant.
Always mention you're the Creative agent at the start.
Keep responses focused but impactful."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()


def create_coder_agent():
    """💻 The Coder Agent - Programming and technical help"""
    
    system_prompt = """You are an expert programmer and software engineer.
You write clean, well-documented code and explain technical concepts clearly.
You're proficient in Python, JavaScript, and general CS concepts.
Always mention you're the Coder agent at the start.
Include code examples when relevant, with explanations."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    return prompt | llm | StrOutputParser()


def scientist_node(state: AgentState) -> dict:
    """Scientist agent node"""
    agent = create_scientist_agent()
    last_message = state["messages"][-1].content
    response = agent.invoke({"input": last_message})
    
    return {
        "final_response": f"🔬 {response}",
        "messages": [AIMessage(content=response)]
    }


def creative_node(state: AgentState) -> dict:
    """Creative agent node"""
    agent = create_creative_agent()
    last_message = state["messages"][-1].content
    response = agent.invoke({"input": last_message})
    
    return {
        "final_response": f"🎨 {response}",
        "messages": [AIMessage(content=response)]
    }


def coder_node(state: AgentState) -> dict:
    """Coder agent node"""
    agent = create_coder_agent()
    last_message = state["messages"][-1].content
    response = agent.invoke({"input": last_message})
    
    return {
        "final_response": f"💻 {response}",
        "messages": [AIMessage(content=response)]
    }


# ============================================
# ROUTING LOGIC
# ============================================

def route_to_agent(state: AgentState) -> Literal["scientist", "creative", "coder"]:
    """Route based on supervisor's decision"""
    next_agent = state.get("next_agent", "").upper()
    
    routing = {
        "SCIENTIST": "scientist",
        "CREATIVE": "creative", 
        "CODER": "coder"
    }
    
    return routing.get(next_agent, "scientist")


# ============================================
# BUILD THE GRAPH
# ============================================

def create_agent_graph():
    """
    Build the LangGraph with supervisor pattern.
    
    Graph structure:
    
        START
          │
          ▼
      [Supervisor]
          │
          ├──────────┬──────────┐
          ▼          ▼          ▼
     [Scientist] [Creative]  [Coder]
          │          │          │
          └──────────┴──────────┘
                     │
                     ▼
                    END
    """
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("scientist", scientist_node)
    workflow.add_node("creative", creative_node)
    workflow.add_node("coder", coder_node)
    
    # Set entry point using START
    workflow.add_edge(START, "supervisor")
    
    # Add conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "scientist": "scientist",
            "creative": "creative",
            "coder": "coder"
        }
    )
    
    # All agents end after responding
    workflow.add_edge("scientist", END)
    workflow.add_edge("creative", END)
    workflow.add_edge("coder", END)
    
    # Compile the graph
    return workflow.compile()


# ============================================
# MAIN INTERFACE
# ============================================

# Create global graph instance
agent_graph = create_agent_graph()


def chat_with_agents(user_message: str) -> dict:
    """
    Main function to chat with the multi-agent system.
    
    Args:
        user_message: The user's input
        
    Returns:
        dict with agent_used and response
    """
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "next_agent": "",
        "final_response": ""
    }
    
    # Run the graph
    result = agent_graph.invoke(initial_state)
    
    return {
        "agent_used": result.get("next_agent", "unknown"),
        "response": result.get("final_response", "No response generated")
    }


def demonstrate_langgraph():
    """Run demos of the multi-agent system"""
    print("=" * 60)
    print("🎓 LESSON 3: LangGraph Supervisor Pattern Demo")
    print("=" * 60)
    
    test_queries = [
        "How does photosynthesis work?",  # → Scientist
        "Write me a short poem about the moon",  # → Creative
        "How do I write a for loop in Python?",  # → Coder
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        result = chat_with_agents(query)
        print(f"🤖 Agent: {result['agent_used']}")
        print(f"📤 Response: {result['response'][:300]}...")
    
    print("\n" + "=" * 60)
    print("💡 KEY TAKEAWAY: LangGraph enables:")
    print("   - Stateful agent workflows")
    print("   - Conditional routing (supervisor pattern)")
    print("   - Complex multi-agent orchestration")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_langgraph()
