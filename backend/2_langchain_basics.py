"""
🎓 LESSON 2: LangChain Abstraction
===================================
LangChain wraps the raw API calls with useful abstractions:
- Chains: Combine multiple steps
- Prompts: Template-based prompt engineering
- Memory: Remember conversation history
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the LLM with OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)


def simple_chain_example(topic: str) -> str:
    """
    A simple LangChain... chain!
    
    Chain = Prompt Template → LLM → Output Parser
    """
    
    # Step 1: Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert teacher who explains concepts simply."),
        ("human", "Explain {topic} in 2-3 sentences for a beginner.")
    ])
    
    # Step 2: Create output parser
    parser = StrOutputParser()
    
    # Step 3: Chain them together using LCEL (LangChain Expression Language)
    chain = prompt | llm | parser
    
    # Step 4: Run the chain
    return chain.invoke({"topic": topic})


def multi_step_chain_example(subject: str) -> dict:
    """
    A multi-step chain that:
    1. Generates a question about a subject
    2. Answers that question
    3. Creates a quiz question
    """
    
    # Chain 1: Generate a question
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", "You generate interesting questions about topics."),
        ("human", "Generate one thought-provoking question about {subject}. Just the question.")
    ])
    question_chain = question_prompt | llm | StrOutputParser()
    
    # Chain 2: Answer the question
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable teacher."),
        ("human", "Answer this question concisely: {question}")
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    # Chain 3: Create a quiz
    quiz_prompt = ChatPromptTemplate.from_messages([
        ("system", "You create educational quiz questions."),
        ("human", "Based on this Q&A, create a multiple choice question:\nQ: {question}\nA: {answer}")
    ])
    quiz_chain = quiz_prompt | llm | StrOutputParser()
    
    # Execute chains sequentially
    question = question_chain.invoke({"subject": subject})
    answer = answer_chain.invoke({"question": question})
    quiz = quiz_chain.invoke({"question": question, "answer": answer})
    
    return {
        "generated_question": question,
        "answer": answer,
        "quiz": quiz
    }


def demonstrate_langchain():
    """Run LangChain demos"""
    print("=" * 60)
    print("🎓 LESSON 2: LangChain Demo")
    print("=" * 60)
    
    # Simple chain
    print("\n📚 Simple Chain Example:")
    print("-" * 40)
    result = simple_chain_example("neural networks")
    print(f"Response: {result}")
    
    # Multi-step chain
    print("\n📚 Multi-Step Chain Example:")
    print("-" * 40)
    result = multi_step_chain_example("machine learning")
    print(f"Generated Question: {result['generated_question']}")
    print(f"Answer: {result['answer']}")
    print(f"Quiz: {result['quiz']}")
    
    print("\n" + "=" * 60)
    print("💡 KEY TAKEAWAY: LangChain provides:")
    print("   - Prompt templates (reusable, parameterized)")
    print("   - Chains (compose multiple steps)")
    print("   - LCEL syntax (prompt | llm | parser)")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_langchain()
