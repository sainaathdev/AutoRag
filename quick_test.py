"""Quick test script for RAG system."""

import os
from pathlib import Path

# Create a simple test document
test_dir = Path("./test_data")
test_dir.mkdir(exist_ok=True)

test_doc = test_dir / "test_document.txt"
test_doc.write_text("""
Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

Neural Networks:
Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information.

Deep Learning:
Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized fields like computer vision and natural language processing.

Applications:
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis
""")

print("✅ Test document created at:", test_doc)
print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("\n1. Ingest the test document:")
print("   python main.py ingest ./test_data")
print("\n2. Query the system:")
print('   python main.py query "What is machine learning?"')
print('   python main.py query "What are types of machine learning?"')
print('   python main.py query "What is deep learning?"')
print("\n3. View statistics:")
print("   python main.py stats")
print("\n4. Launch dashboard:")
print("   python main.py dashboard")
print("\n" + "="*60)
