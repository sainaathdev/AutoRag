# Example usage script for Self-Improving RAG System

from rag_system import SelfImprovingRAG
from pathlib import Path


def example_basic_usage():
    """Basic usage example."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize system
    rag = SelfImprovingRAG()
    
    # Create sample document
    sample_doc = Path("sample_document.txt")
    sample_doc.write_text("""
    Machine Learning Basics
    
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. It focuses on the
    development of computer programs that can access data and use it to learn for themselves.
    
    Types of Machine Learning:
    
    1. Supervised Learning: The algorithm learns from labeled training data and makes
    predictions based on that data. Examples include classification and regression.
    
    2. Unsupervised Learning: The algorithm learns from unlabeled data and finds hidden
    patterns or intrinsic structures. Examples include clustering and dimensionality reduction.
    
    3. Reinforcement Learning: The algorithm learns through interaction with an environment,
    receiving rewards or penalties for actions taken.
    
    Neural Networks:
    
    Neural networks are computing systems inspired by biological neural networks. They consist
    of interconnected nodes (neurons) organized in layers. Deep learning uses neural networks
    with multiple hidden layers to learn complex patterns in data.
    
    Applications:
    
    - Image recognition and computer vision
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    """)
    
    # Ingest document
    print("\nIngesting sample document...")
    chunk_count = rag.ingest_document(str(sample_doc))
    print(f"✓ Created {chunk_count} chunks")
    
    # Query the system
    print("\nQuerying the system...")
    result = rag.query(
        "What are the types of machine learning?",
        return_metadata=True
    )
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence_score']:.2%}")
    
    # Clean up
    sample_doc.unlink()


def example_advanced_usage():
    """Advanced usage with multiple queries."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced Usage")
    print("="*80)
    
    rag = SelfImprovingRAG()
    
    # Multiple queries to demonstrate learning
    queries = [
        "What is supervised learning?",
        "Explain neural networks",
        "What are applications of machine learning?",
        "How does reinforcement learning work?",
        "What is deep learning?"
    ]
    
    print("\nProcessing multiple queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        result = rag.query(query)
        print(f"Confidence: {result['confidence_score']:.2%}")
    
    # Show statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    
    stats = rag.get_statistics()
    
    print(f"\nTotal Queries: {stats['feedback']['total_queries']}")
    print(f"Average Confidence: {stats['confidence']['current_avg']:.2%}")
    print(f"Confidence Trend: {stats['confidence']['trend']}")
    print(f"Failure Rate: {stats['feedback']['failure_rate']:.1%}")


def example_optimization():
    """Example showing optimization features."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Optimization Features")
    print("="*80)
    
    rag = SelfImprovingRAG()
    
    # Get optimization report
    opt_report = rag.optimizer.get_optimization_report()
    
    print("\nOptimization Report:")
    print(f"Best Retrieval Method: {opt_report['best_retrieval_method']}")
    print(f"Problematic Documents: {len(opt_report['problematic_documents'])}")
    print(f"Total Queries Tracked: {opt_report['total_queries_tracked']}")
    
    # Show retrieval method performance
    print("\nRetrieval Method Performance:")
    for method, stats in opt_report['retrieval_methods'].items():
        print(f"  {method.capitalize()}:")
        print(f"    Queries: {stats['queries']}")
        print(f"    Avg Confidence: {stats['avg_confidence']:.2%}")


if __name__ == "__main__":
    print("\n🧠 Self-Improving RAG System - Examples\n")
    
    try:
        example_basic_usage()
        example_advanced_usage()
        example_optimization()
        
        print("\n" + "="*80)
        print("✓ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set DEEPSEEK_API_KEY in .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")
