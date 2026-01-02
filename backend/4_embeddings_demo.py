"""
🎓 LESSON 4: Understanding Embeddings
======================================
Embeddings are the secret sauce of modern AI!

They convert text → numbers (vectors) that capture MEANING.
Similar meanings = similar vectors = close in "embedding space"

This lesson teaches:
1. What embeddings are
2. How to generate them
3. Semantic similarity
4. Practical applications (search, clustering)
"""

import os
import httpx
import numpy as np
from typing import List
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_embedding(text: str, model: str = "openai/text-embedding-3-small") -> List[float]:
    """
    Convert text to an embedding vector.
    
    This is the fundamental operation - text goes in, numbers come out!
    
    Args:
        text: The text to embed
        model: The embedding model to use
        
    Returns:
        A list of floats (the embedding vector)
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
    }
    
    payload = {
        "model": model,
        "input": text
    }
    
    with httpx.Client() as client:
        response = client.post(
            "https://openrouter.ai/api/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
    
    result = response.json()
    return result["data"][0]["embedding"]


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the angle between vectors:
    - 1.0 = identical direction (same meaning)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (opposite meaning)
    """
    
    # Reshape for sklearn
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    
    return cosine_similarity(v1, v2)[0][0]


def demonstrate_basic_embeddings():
    """Demo 1: Basic embedding generation"""
    print("\n📊 Demo 1: Basic Embeddings")
    print("-" * 40)
    
    text = "Machine learning is fascinating"
    embedding = get_embedding(text)
    
    print(f"Text: '{text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")


def demonstrate_semantic_similarity():
    """Demo 2: Semantic similarity"""
    print("\n📊 Demo 2: Semantic Similarity")
    print("-" * 40)
    
    # Test pairs - some similar, some different
    test_pairs = [
        ("I love programming", "Coding is my passion"),  # Similar meaning
        ("I love programming", "I hate programming"),     # Opposite sentiment
        ("I love programming", "The weather is nice"),    # Unrelated
        ("Python is great", "Python is an excellent language"),  # Very similar
        ("cat", "kitten"),  # Related concepts
        ("cat", "automobile"),  # Unrelated concepts
    ]
    
    print("\nComparing text pairs:\n")
    
    for text1, text2 in test_pairs:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        similarity = calculate_similarity(emb1, emb2)
        
        # Visual indicator
        bar_length = int(similarity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"'{text1}' vs '{text2}'")
        print(f"  Similarity: {similarity:.4f} [{bar}]")
        print()


def demonstrate_semantic_search():
    """Demo 3: Semantic Search"""
    print("\n📊 Demo 3: Semantic Search (Finding Similar Documents)")
    print("-" * 40)
    
    # Our "document database"
    documents = [
        "Python is a versatile programming language",
        "JavaScript runs in web browsers",
        "Machine learning models learn from data",
        "Cats are independent pets",
        "Dogs are loyal companions",
        "Neural networks are inspired by the brain",
        "The sun is a star at the center of our solar system",
    ]
    
    # Pre-compute embeddings for all documents
    print("Indexing documents...")
    doc_embeddings = [get_embedding(doc) for doc in documents]
    
    # Search queries
    queries = [
        "How do I code in Python?",
        "Tell me about pets",
        "Deep learning AI"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        query_embedding = get_embedding(query)
        
        # Calculate similarity to all documents
        similarities = [
            calculate_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]
        
        # Rank by similarity
        ranked = sorted(
            zip(documents, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("   Top 3 results:")
        for i, (doc, score) in enumerate(ranked[:3], 1):
            print(f"   {i}. [{score:.3f}] {doc}")


def demonstrate_clustering_concept():
    """Demo 4: Conceptual Clustering"""
    print("\n📊 Demo 4: Conceptual Clustering")
    print("-" * 40)
    
    # Words from different categories
    words = {
        "animals": ["cat", "dog", "elephant", "lion"],
        "fruits": ["apple", "banana", "orange", "grape"],
        "tech": ["computer", "software", "algorithm", "database"]
    }
    
    # Flatten and get embeddings
    all_words = []
    labels = []
    for category, word_list in words.items():
        all_words.extend(word_list)
        labels.extend([category] * len(word_list))
    
    print("Getting embeddings for words...")
    embeddings = [get_embedding(word) for word in all_words]
    
    # Show within-category vs between-category similarity
    print("\n📈 Average similarities:")
    
    for cat1 in words.keys():
        for cat2 in words.keys():
            if cat1 <= cat2:  # Avoid duplicates
                idx1 = [i for i, l in enumerate(labels) if l == cat1]
                idx2 = [i for i, l in enumerate(labels) if l == cat2]
                
                similarities = []
                for i in idx1:
                    for j in idx2:
                        if i != j:
                            sim = calculate_similarity(embeddings[i], embeddings[j])
                            similarities.append(sim)
                
                avg_sim = np.mean(similarities)
                print(f"   {cat1} ↔ {cat2}: {avg_sim:.3f}")


def run_all_demos():
    """Run all embedding demonstrations"""
    print("=" * 60)
    print("🎓 LESSON 4: Understanding Embeddings")
    print("=" * 60)
    
    demonstrate_basic_embeddings()
    demonstrate_semantic_similarity()
    demonstrate_semantic_search()
    demonstrate_clustering_concept()
    
    print("\n" + "=" * 60)
    print("💡 KEY TAKEAWAYS:")
    print("   - Embeddings convert text → vectors")
    print("   - Similar meanings = similar vectors")
    print("   - Enables: search, clustering, recommendations")
    print("   - Foundation of RAG (Retrieval Augmented Generation)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()

