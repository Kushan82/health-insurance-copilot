"""
Test Advanced RAG Features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.rag.query_rewriter import QueryRewriter, MultiQueryGenerator, QueryExpander
from loguru import logger


def test_query_rewriter():
    print("\n" + "="*60)
    print("  🔄 TESTING QUERY REWRITER")
    print("="*60)
    
    rewriter = QueryRewriter()
    
    test_queries = [
        "What is PED waiting?",
        "Does optima cover maturnity?",
        "Room rent sub-limit in medicare",
        "AYUSH treatment coverage",
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        result = rewriter.rewrite(query)
        print(f"✅ Rewritten: {result['rewritten']}")
        print(f"🔀 Expansions: {result['expansions']}")


def test_multi_query_generator():
    print("\n" + "="*60)
    print("  🔢 TESTING MULTI-QUERY GENERATOR")
    print("="*60)
    
    generator = MultiQueryGenerator()
    
    test_queries = [
        "What is the waiting period for pre-existing diseases?",
        "Does Care Supreme cover maternity?",
        "Room rent limits in Medicare Plus",
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        variations = generator.generate(query)
        print(f"✅ Generated {len(variations)} variations:")
        for i, var in enumerate(variations, 1):
            print(f"   {i}. {var}")


def test_query_expander():
    print("\n" + "="*60)
    print("  📈 TESTING QUERY EXPANDER")
    print("="*60)
    
    expander = QueryExpander()
    
    test_queries = [
        "What is the waiting period for pre-existing disease?",
        "Does policy cover room rent?",
        "Tell me about maternity benefits",
    ]
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        result = expander.expand(query)
        print(f"✅ Expanded terms: {result['expanded_terms']}")
        print(f"🔀 Variants:")
        for var in result['all_variants']:
            print(f"   • {var}")


if __name__ == "__main__":
    test_query_rewriter()
    test_multi_query_generator()
    test_query_expander()
    
    print("\n" + "="*60)
    print("  ✅ ALL TESTS COMPLETE!")
    print("="*60)
