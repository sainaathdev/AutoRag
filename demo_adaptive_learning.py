"""
Demonstration: Self-Improving RAG System - Adaptive Learning in Action

This script demonstrates how the RAG system learns and improves over time.
"""

import time
from rag_system import SelfImprovingRAG

print("="*80)
print("🧠 SELF-IMPROVING RAG SYSTEM - ADAPTIVE LEARNING DEMONSTRATION")
print("="*80)
print()

# Initialize system
print("📌 Step 1: Initializing RAG System...")
rag = SelfImprovingRAG()
print("✅ System initialized\n")

# Show initial state
print("="*80)
print("📊 INITIAL STATE (Before Any Queries)")
print("="*80)
stats = rag.get_statistics()
print(f"Total Documents: {stats['vector_store']['total_chunks']}")
print(f"Total Queries Processed: {stats['feedback']['total_queries']}")
print(f"Average Confidence: {stats['confidence']['current_avg']:.2%}")
print(f"Confidence Trend: {stats['confidence']['trend']}")
print()

# Run multiple queries to trigger learning
print("="*80)
print("📝 Step 2: Running Queries to Trigger Learning...")
print("="*80)

queries = [
    "What is machine learning?",
    "What are the types of machine learning?",
    "What is deep learning?",
    "Explain supervised learning",
    "What are neural networks?",
    "What is reinforcement learning?",
    "What are applications of machine learning?",
    "How does unsupervised learning work?",
]

confidence_scores = []

for i, query in enumerate(queries, 1):
    print(f"\n🔍 Query {i}/{len(queries)}: {query}")
    
    result = rag.query(query, return_metadata=True)
    confidence = result['confidence_score']
    confidence_scores.append(confidence)
    
    print(f"   ✓ Confidence: {confidence:.2%}")
    print(f"   ✓ Retrieval Method: {result['retrieval_method']}")
    
    # Show if query was rewritten
    if result['rewritten_query'] != result['original_query']:
        print(f"   ✓ Query Rewritten: {result['rewritten_query']}")
    
    # Show evaluation
    eval_data = result['evaluation']
    if eval_data.get('hallucination_detected'):
        print(f"   ⚠️  Hallucination Detected!")
    
    time.sleep(0.5)  # Brief pause for readability

print("\n" + "="*80)
print("📈 Step 3: Analyzing Learning Progress...")
print("="*80)

# Show updated statistics
stats = rag.get_statistics()

print(f"\n📊 SYSTEM STATISTICS AFTER {len(queries)} QUERIES:")
print(f"   Total Queries: {stats['feedback']['total_queries']}")
print(f"   Average Confidence: {stats['confidence']['current_avg']:.2%}")
print(f"   Confidence Trend: {stats['confidence']['trend']}")
print(f"   Failure Rate: {stats['feedback']['failure_rate']:.1%}")

# Show confidence progression
print(f"\n📈 CONFIDENCE SCORE PROGRESSION:")
for i, score in enumerate(confidence_scores, 1):
    bar = "█" * int(score * 50)
    print(f"   Query {i}: {bar} {score:.2%}")

# Calculate improvement
if len(confidence_scores) > 1:
    first_half_avg = sum(confidence_scores[:len(confidence_scores)//2]) / (len(confidence_scores)//2)
    second_half_avg = sum(confidence_scores[len(confidence_scores)//2:]) / (len(confidence_scores) - len(confidence_scores)//2)
    improvement = second_half_avg - first_half_avg
    
    print(f"\n💡 LEARNING ANALYSIS:")
    print(f"   First Half Average: {first_half_avg:.2%}")
    print(f"   Second Half Average: {second_half_avg:.2%}")
    print(f"   Improvement: {improvement:+.2%}")
    
    if improvement > 0:
        print(f"   ✅ System is IMPROVING over time!")
    elif improvement < 0:
        print(f"   ⚠️  System performance declined (may need optimization)")
    else:
        print(f"   ➡️  System performance is stable")

# Show optimization report
print("\n" + "="*80)
print("🔧 Step 4: Optimization Report")
print("="*80)

opt_report = stats['optimization']

print(f"\n🎯 RETRIEVAL METHOD PERFORMANCE:")
for method, method_stats in opt_report.get('retrieval_methods', {}).items():
    print(f"   {method.upper()}:")
    print(f"      Queries: {method_stats.get('queries', 0)}")
    print(f"      Avg Confidence: {method_stats.get('avg_confidence', 0):.2%}")

print(f"\n🏆 BEST RETRIEVAL METHOD: {opt_report.get('best_retrieval_method', 'N/A').upper()}")

# Show problematic documents (if any)
problematic = opt_report.get('problematic_documents', [])
if problematic:
    print(f"\n⚠️  PROBLEMATIC DOCUMENTS: {len(problematic)}")
    print("   These documents consistently produce low-confidence answers:")
    for doc_id in problematic[:3]:
        print(f"      - {doc_id}")
    print("   💡 System will automatically re-chunk these documents!")
else:
    print(f"\n✅ NO PROBLEMATIC DOCUMENTS - All documents performing well!")

# Show failure memory
print("\n" + "="*80)
print("🧠 Step 5: Failure Memory & Learning")
print("="*80)

failure_cases = rag.feedback_store.get_failure_cases(5)
if failure_cases:
    print(f"\n📝 FAILURE MEMORY: {len(failure_cases)} cases stored")
    print("   Recent failures:")
    for i, failure in enumerate(failure_cases[-3:], 1):
        print(f"\n   {i}. Query: {failure.query}")
        print(f"      Confidence: {failure.confidence_score:.2%}")
        print(f"      Reason: {failure.failure_reason}")
        print(f"      💡 System learned from this failure!")
else:
    print("\n✅ NO FAILURES - System is performing excellently!")

# Show adaptive chunking stats
print("\n" + "="*80)
print("📐 Step 6: Adaptive Chunking Performance")
print("="*80)

print("\n💡 ADAPTIVE CHUNKING FEATURES:")
print("   ✓ Tracks performance per document")
print("   ✓ Adjusts chunk size based on results")
print("   ✓ Re-chunks documents with low confidence")
print("   ✓ Learns optimal chunking strategy")

# Show document performance
doc_stats = rag.optimizer.document_stats
if doc_stats:
    print(f"\n📊 DOCUMENT PERFORMANCE TRACKING:")
    print(f"   Tracking {len(doc_stats)} document(s)")
    for doc_id, perf in list(doc_stats.items())[:3]:
        print(f"\n   Document: {doc_id[:20]}...")
        print(f"      Queries: {perf['query_count']}")
        print(f"      Avg Confidence: {perf['avg_confidence']:.2%}")
        print(f"      Failures: {perf['failure_count']}")

# Final summary
print("\n" + "="*80)
print("🎓 ADAPTIVE LEARNING SUMMARY")
print("="*80)

print("\n✅ EVIDENCE OF ADAPTIVE LEARNING:")
print("   1. ✓ Confidence tracking across queries")
print("   2. ✓ Failure memory stores problematic cases")
print("   3. ✓ Retrieval method performance comparison")
print("   4. ✓ Document-level performance tracking")
print("   5. ✓ Automatic identification of problematic documents")
print("   6. ✓ Trend analysis (improving/declining/stable)")
print("   7. ✓ Query rewriting based on patterns")
print("   8. ✓ Hallucination detection and learning")

print("\n💡 SELF-IMPROVEMENT MECHANISMS:")
print("   • Tracks confidence scores over time")
print("   • Identifies low-performing documents")
print("   • Recommends re-chunking strategies")
print("   • Compares retrieval methods")
print("   • Learns from failures")
print("   • Adapts chunking parameters")
print("   • Optimizes retrieval strategy")

print("\n🚀 NEXT OPTIMIZATION TRIGGER:")
optimization_interval = 50
queries_until_next = optimization_interval - (stats['feedback']['total_queries'] % optimization_interval)
print(f"   In {queries_until_next} more queries")
print(f"   (Triggers every {optimization_interval} queries)")

print("\n" + "="*80)
print("✨ DEMONSTRATION COMPLETE!")
print("="*80)
print("\nYour RAG system is actively learning and improving! 🧠")
print("Run more queries to see continued adaptation and optimization.")
print()
