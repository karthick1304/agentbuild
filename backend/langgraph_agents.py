"""
LangGraph Agents Module
========================
Multi-agent system with supervisor pattern using LangGraph.
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
    model="qwen/qwen-2.5-72b-instruct",  # Free on OpenRouter!
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost:3000"}
)


class AgentState(TypedDict):
    """The state that flows through our graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    final_response: str


def create_supervisor():
    """The Supervisor decides which specialist should handle the query."""
    
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
    last_message = state["messages"][-1].content
    decision = supervisor.invoke({"input": last_message}).strip().upper()
    
    valid_agents = ["SCIENTIST", "CREATIVE", "CODER"]
    if decision not in valid_agents:
        decision = "SCIENTIST"
    
    print(f"🎯 Supervisor Decision: Route to {decision}")
    
    return {"next_agent": decision}


def create_scientist_agent():
    """🔬 The Scientist Agent"""
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
    """🎨 The Creative Agent"""
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
    """💻 The Coder Agent"""
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


def route_to_agent(state: AgentState) -> Literal["scientist", "creative", "coder"]:
    """Route based on supervisor's decision"""
    next_agent = state.get("next_agent", "").upper()
    routing = {
        "SCIENTIST": "scientist",
        "CREATIVE": "creative", 
        "CODER": "coder"
    }
    return routing.get(next_agent, "scientist")


def create_agent_graph():
    """Build the LangGraph with supervisor pattern."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("scientist", scientist_node)
    workflow.add_node("creative", creative_node)
    workflow.add_node("coder", coder_node)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "scientist": "scientist",
            "creative": "creative",
            "coder": "coder"
        }
    )
    
    workflow.add_edge("scientist", END)
    workflow.add_edge("creative", END)
    workflow.add_edge("coder", END)
    
    return workflow.compile()


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
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "next_agent": "",
        "final_response": ""
    }
    
    result = agent_graph.invoke(initial_state)
    
    return {
        "agent_used": result.get("next_agent", "unknown"),
        "response": result.get("final_response", "No response generated")
    }
