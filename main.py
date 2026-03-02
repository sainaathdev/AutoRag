"""Main entry point for the Self-Improving RAG System."""

import argparse
from pathlib import Path

from rag_system import SelfImprovingRAG
from utils.logger import setup_logger


logger = setup_logger(__name__)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Self-Improving RAG System")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", type=str, help="Path to document or directory")
    ingest_parser.add_argument("--recursive", action="store_true", help="Process subdirectories")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch dashboard")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = SelfImprovingRAG(config_path=args.config)
    
    if args.command == "ingest":
        path = Path(args.path)
        
        if not path.exists():
            logger.error(f"Path not found: {args.path}")
            return
        
        if path.is_file():
            count = rag.ingest_document(str(path))
            print(f"✓ Ingested {count} chunks from {path.name}")
        else:
            count = rag.ingest_directory(str(path), recursive=args.recursive)
            print(f"✓ Ingested {count} total chunks from {path.name}")
    
    elif args.command == "query":
        result = rag.query(args.query, top_k=args.top_k, return_metadata=args.verbose)
        
        print("\n" + "="*80)
        print("ANSWER")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        print(f"Confidence: {result['confidence_score']:.2%}")
        print("="*80)
        
        if args.verbose:
            print("\nRETRIEVED DOCUMENTS:")
            for i, chunk in enumerate(result["retrieved_chunks"]):
                print(f"\n[Document {i+1}]")
                print(chunk["text"][:300] + "...")
    
    elif args.command == "stats":
        stats = rag.get_statistics()
        
        print("\n" + "="*80)
        print("SYSTEM STATISTICS")
        print("="*80)
        
        print("\nVector Store:")
        for key, value in stats["vector_store"].items():
            print(f"  {key}: {value}")
        
        print("\nConfidence:")
        for key, value in stats["confidence"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        print("\nFeedback:")
        for key, value in stats["feedback"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
        
        print("="*80)
    
    elif args.command == "dashboard":
        import subprocess
        import sys
        
        print("Launching dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
