"""
Test retrieval precision on a document with many similar paragraphs.

This test evaluates whether the system can distinguish between 30 highly similar
paragraphs about utilitarianism variants and retrieve the specific one that
answers each query.

Key challenge: All paragraphs discuss utilitarianism and share similar vocabulary,
but each describes a distinct variant. The cross-encoder should excel at finding
the exact paragraph with the specific detail being asked about.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import CrossEncoderReranker
import math
from typing import Dict, List, Tuple


# Test queries with graded relevance judgments (0-3 scale)
# 3 = Highly relevant (directly answers with specific details)
# 1 = Marginally relevant (mentions related concepts)
# 0 = Not relevant
#
TEST_QUERIES = [
    # Paragraph 0: General introduction
    {
        "query": "Over what timeframe has the process of refining utility developed?",
        "relevance_judgments": {0: 3, 1: 1},
        "description": "Abstracted detail 'centuries' (Linked to 'late 18th century' in Para 1)",
    },
    # Paragraph 1: Classical utilitarianism (Bentham)
    {
        "query": "How do you measure pain versus pleasure?",
        "relevance_judgments": {1: 3, 37: 1},
        "description": "Target detail 'hedonic calculus' (Linked to Cardinal util in Para 37)",
    },
    # Paragraph 2: Mill's qualitative utilitarianism
    {
        "query": "Is it better to be a dissatisfied human or a satisfied fool?",
        "relevance_judgments": {2: 3, 30: 1},
        "description": "Target detail 'Socrates' (Linked to Ideal util/higher values in Para 30)",
    },
    # Paragraph 3: Act utilitarianism
    {
        "query": "Do general rules matter when judging specific actions?",
        "relevance_judgments": {3: 3, 45: 1},
        "description": "Abstracted detail 'without reference to rules' (Linked to Single-level in Para 45)",
    },
    # Paragraph 4: Rule utilitarianism
    {
        "query": "What is the goal of following set guidelines?",
        "relevance_judgments": {4: 3, 23: 1},
        "description": "Abstracted detail 'promote the greatest good' (Linked to Rule-consequentialism in Para 23)",
    },
    # Paragraph 5: Average utilitarianism
    {
        "query": "Is a large low-happiness population preferred?",
        "relevance_judgments": {5: 3, 6: 1},
        "description": "Abstracted population preference (Contrast with Total util in Para 6)",
    },
    # Paragraph 6: Total utilitarianism
    {
        "query": "What paradox involves a large barely happy population?",
        "relevance_judgments": {6: 3, 33: 1},
        "description": "Target detail 'repugnant conclusion' (Linked to Impersonal util in Para 33)",
    },
    # Paragraph 7: Preference utilitarianism
    {
        "query": "Who helped develop the autonomy-focused ethical approach?",
        "relevance_judgments": {7: 3, 9: 1},
        "description": "Target detail 'R.M. Hare' (Linked to Two-level util in Para 9)",
    },
    # Paragraph 8: Negative utilitarianism
    {
        "query": "In suffering-focused ethics, is pain rated more intense than pleasure?",
        "relevance_judgments": {8: 3, 44: 1},
        "description": "Target detail 'suffering is more intense' (Linked to Necessitarian util in Para 44)",
    },
    # Paragraph 9: Two-level utilitarianism
    {
        "query": "When does intuitive thinking need to switch to calculation?",
        "relevance_judgments": {9: 3, 20: 1},
        "description": "Target detail 'when rules conflict' (Linked to Multi-level util in Para 20)",
    },
    # Paragraph 10: Motive utilitarianism
    {
        "query": "Is good character valued even with bad results?",
        "relevance_judgments": {10: 3, 28: 1},
        "description": "Abstracted 'fail to maximize' (Linked to Virtue consequentialism in Para 28)",
    },
    # Paragraph 11: Global utilitarianism
    {
        "query": "Which modern philanthropy movement uses cosmopolitan moral arguments?",
        "relevance_judgments": {11: 3, 41: 1},
        "description": "Target detail 'effective altruism' (Linked to Sentientist util in Para 41)",
    },
    # Paragraph 12: Hedonistic utilitarianism
    {
        "query": "What theory focuses only on pleasure?",
        "relevance_judgments": {12: 3, 1: 1},
        "description": "Target detail 'monistic' (Linked to Classical util in Para 1)",
    },
    # Paragraph 13: Welfare utilitarianism
    {
        "query": "Does well-being have multiple factors?",
        "relevance_judgments": {13: 3, 26: 1},
        "description": "Target detail 'pluralistic' (Linked to Objective List in Para 26)",
    },
    # Paragraph 14: Scalar utilitarianism
    {
        "query": "If utility is a continuum, are there sharp moral boundaries?",
        "relevance_judgments": {14: 3, 15: 1},
        "description": "Target detail 'no sharp boundaries' (Contrast with Satisficing in Para 15)",
    },
    # Paragraph 15: Satisficing utilitarianism
    {
        "query": "What problem does the 'good enough' approach fix?",
        "relevance_judgments": {15: 3, 3: 1},
        "description": "Target detail 'demanding nature' (Linked to Act util in Para 3)",
    },
    # Paragraph 16: Cooperative utilitarianism
    {
        "query": "Can cooperation produce better results than individuals?",
        "relevance_judgments": {16: 3, 48: 1},
        "description": "Abstracted 'better overall outcomes' (Linked to Universal util in Para 48)",
    },
    # Paragraph 17: Indirect utilitarianism
    {
        "query": "If avoiding direct calculation, what guides ensure good results?",
        "relevance_judgments": {17: 3, 4: 1},
        "description": "Target detail 'virtues, rules, or dispositions' (Linked to Rule util in Para 4)",
    },
    # Paragraph 18: Generational utilitarianism
    {
        "query": "Is the future value of unborn people less?",
        "relevance_judgments": {18: 3, 32: 1},
        "description": "Target detail 'temporal discounting' (Contrast with Person-affecting in Para 32)",
    },
    # Paragraph 19: Expected utility utilitarianism
    {
        "query": "Why prioritize probability over actual results?",
        "relevance_judgments": {19: 3, 25: 1},
        "description": "Abstracted 'rarely know with certainty' (Linked to Foreseeable conseq in Para 25)",
    },
    # Paragraph 20: Multi-level utilitarianism
    {
        "query": "In multi-level systems, do laws differ from individual acts?",
        "relevance_judgments": {20: 3, 40: 1},
        "description": "Target detail 'constitutional principles' (Linked to Institutional util in Para 40)",
    },
    # Paragraph 21: Prioritarian utilitarianism
    {
        "query": "Is distribution part of maximizing welfare?",
        "relevance_judgments": {21: 3, 39: 1},
        "description": "Confirmation 'focused on maximizing total welfare' (Linked to Geometric util in Para 39)",
    },
    # Paragraph 22: Threshold utilitarianism
    {
        "query": "What non-utility factor is combined with maximization for a baseline?",
        "relevance_judgments": {22: 3, 46: 1},
        "description": "Target detail 'rights-based' (Linked to Hierarchical util in Para 46)",
    },
    # Paragraph 23: Rule-consequentialism
    {
        "query": "When assessing rules, can values other than utility be included?",
        "relevance_judgments": {23: 3, 4: 1},
        "description": "Abstracted 'incorporate non-utilitarian values' (Contrast with Rule util in Para 4)",
    },
    # Paragraph 24: Actual consequence utilitarianism
    {
        "query": "How does luck affect judging actions by results?",
        "relevance_judgments": {24: 3, 25: 1},
        "description": "Target detail 'moral luck' (Contrast with Foreseeable in Para 25)",
    },
    # Paragraph 25: Foreseeable consequence utilitarianism
    {
        "query": "Does judging by likelihood help maintain agent responsibility?",
        "relevance_judgments": {25: 3, 19: 1},
        "description": "Abstracted 'holding agents responsible' (Linked to Expected util in Para 19)",
    },
    # Paragraph 26: Objective list utilitarianism
    {
        "query": "In a checklist of well-being, which social bond is a good?",
        "relevance_judgments": {26: 3, 30: 1},
        "description": "Target detail 'friendship' (Linked to Ideal util in Para 30)",
    },
    # Paragraph 27: Desire-fulfillment utilitarianism
    {
        "query": "Can unknown events provide satisfaction?",
        "relevance_judgments": {27: 3, 7: 1},
        "description": "Abstracted 'don't directly affect experiences' (Linked to Preference util in Para 7)",
    },
    # Paragraph 28: Virtue consequentialism
    {
        "query": "Do virtues always yield good outcomes?",
        "relevance_judgments": {28: 3, 10: 1},
        "description": "Abstracted 'dispositions that generally promote good' (Linked to Motive util in Para 10)",
    },
    # Paragraph 29: Restricted utilitarianism
    {
        "query": "Which public domain is best for calculation?",
        "relevance_judgments": {29: 3, 40: 1},
        "description": "Target detail 'policy' (Linked to Institutional util in Para 40)",
    },
    # Paragraph 30: Ideal utilitarianism
    {
        "query": "Can things have value without causing happiness?",
        "relevance_judgments": {30: 3, 26: 1},
        "description": "Abstracted 'do not produce positive mental states' (Linked to Objective List in Para 26)",
    },
    # Paragraph 31: Theological utilitarianism
    {
        "query": "Is promoting utility a duty backed by God?",
        "relevance_judgments": {31: 3, 1: 1},
        "description": "Target detail 'divine sanctions' (Linked to Classical/Paley context in Para 1)",
    },
    # Paragraph 32: Person-affecting utilitarianism
    {
        "query": "Is there a need to maximize population?",
        "relevance_judgments": {32: 3, 6: 1},
        "description": "Rejection of 'maximize population size' (Contrast with Total util in Para 6)",
    },
    # Paragraph 33: Impersonal utilitarianism
    {
        "query": "Is creating new happy beings good?",
        "relevance_judgments": {33: 3, 6: 1},
        "description": "Target detail 'creating new happy beings' (Linked to Total util in Para 6)",
    },
    # Paragraph 34: Subjective utilitarianism
    {
        "query": "Is an action right if you believe it works?",
        "relevance_judgments": {34: 3, 25: 1},
        "description": "Abstracted 'beliefs and perspective' (Linked to Foreseeable in Para 25)",
    },
    # Paragraph 35: Objective utilitarianism
    {
        "query": "Is failure wrong despite good intentions?",
        "relevance_judgments": {35: 3, 24: 1},
        "description": "Abstracted 'fails to produce best possible outcome' (Linked to Actual consequence in Para 24)",
    },
    # Paragraph 36: Ordinal utilitarianism
    {
        "query": "Does the measurement allow one outcome to be rated 'twice as good'?",
        "relevance_judgments": {36: 3, 37: 1},
        "description": "Rejection of 'twice as good' (Contrast with Cardinal util in Para 37)",
    },
    # Paragraph 37: Cardinal utilitarianism
    {
        "query": "Can total welfare be calculated mathematically?",
        "relevance_judgments": {37: 3, 1: 1},
        "description": "Abstracted 'summation of total welfare' (Linked to Classical util in Para 1)",
    },
    # Paragraph 38: Critical-level utilitarianism
    {
        "query": "Do lives barely worth living add value?",
        "relevance_judgments": {38: 3, 6: 1},
        "description": "Rejection of 'barely worth living' (Contrast with Repugnant Conclusion in Para 6)",
    },
    # Paragraph 39: Geometric utilitarianism
    {
        "query": "Do resources help the wealthy less?",
        "relevance_judgments": {39: 3, 21: 1},
        "description": "Target detail 'wealthy person vs poor person' (Linked to Prioritarian in Para 21)",
    },
    # Paragraph 40: Institutional utilitarianism
    {
        "query": "Must public structures be impartial?",
        "relevance_judgments": {40: 3, 20: 1},
        "description": "Abstracted 'institutions must be impartial' (Linked to Multi-level util in Para 20)",
    },
    # Paragraph 41: Sentientist utilitarianism
    {
        "query": "Do non-living entities have value?",
        "relevance_judgments": {41: 3, 11: 1},
        "description": "Rejection of 'non-living entities' (Linked to Global util in Para 11)",
    },
    # Paragraph 42: Anthropocentric utilitarianism
    {
        "query": "Is animal suffering relevant only for its impact on humans?",
        "relevance_judgments": {42: 3, 41: 1},
        "description": "Target detail 'instrumentally relevant' (Contrast with Sentientist in Para 41)",
    },
    # Paragraph 43: Liberal utilitarianism
    {
        "query": "Is a sphere of personal freedom essential for long-term utility?",
        "relevance_judgments": {43: 3, 2: 1},
        "description": "Target detail 'sphere of protected rights' (Linked to Mill/Liberty in Para 2)",
    },
    # Paragraph 44: Necessitarian utilitarianism
    {
        "query": "Is there an obligation to create happy beings?",
        "relevance_judgments": {44: 3, 32: 1},
        "description": "Rejection of 'obligation to bring into existence' (Linked to Person-affecting in Para 32)",
    },
    # Paragraph 45: Single-level utilitarianism
    {
        "query": "In direct maximization, are rules of thumb dangerous?",
        "relevance_judgments": {45: 3, 3: 1},
        "description": "Target detail 'dangerous distractions' (Linked to Act util in Para 3)",
    },
    # Paragraph 46: Hierarchical utilitarianism
    {
        "query": "Do basic needs take strict priority over luxuries?",
        "relevance_judgments": {46: 3, 22: 1},
        "description": "Target detail 'lexical priority' (Linked to Threshold util in Para 22)",
    },
    # Paragraph 47: Evolutionary utilitarianism
    {
        "query": "Are moral senses shortcuts for utility?",
        "relevance_judgments": {47: 3, 17: 1},
        "description": "Target detail 'heuristic for utility' (Linked to Indirect util in Para 17)",
    },
    # Paragraph 48: Universal utilitarianism
    {
        "query": "Does the universal approach bridge duty and consequences?",
        "relevance_judgments": {48: 3, 4: 1},
        "description": "Target detail 'bridging Kantian and utilitarian' (Linked to Rule util in Para 4)",
    },
    # Paragraph 49: Agent-relative utilitarianism
    {
        "query": "Does the relative view allow for personal commitments?",
        "relevance_judgments": {49: 3, 40: 1},
        "description": "Abstracted 'accommodate personal commitments' (Contrast with Institutional in Para 40)",
    },
]


def ingest_document():
    """Ingest the philosophy document into the vector store."""
    print("Ingesting eval_philosophy_document.txt...")

    # Read document
    doc_path = os.path.join(os.path.dirname(__file__), "eval_philosophy_document.txt")
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize services
    embedding_service = EmbeddingService()

    # Clear existing data to ensure clean test
    import shutil

    data_dir = "data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    vector_store = VectorStore()

    # Process document
    from app.utils.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    sentences, paragraph_indices = processor.split_into_sentences_and_paragraphs(text)

    print(
        f"Found {len(sentences)} sentences across {max(paragraph_indices) + 1} paragraphs"
    )

    # Generate embeddings
    embeddings = embedding_service.embed_texts(sentences)

    # Add to vector store
    vector_store.index_document(
        doc_id="philosophy_util",
        filename="eval_philosophy_document.txt",
        file_type=".txt",
        sentences=sentences,
        embeddings=embeddings,
        paragraph_indices=paragraph_indices,
    )

    print(f"✓ Ingested document with {len(sentences)} sentences\n")
    return vector_store


def calculate_ndcg(
    retrieved_paragraphs: List[int], relevance_judgments: Dict[int, int], k: int = 10
) -> float:
    """
    Calculate nDCG@k for a single query with graded relevance.

    Args:
        retrieved_paragraphs: List of retrieved paragraph indices in ranked order
        relevance_judgments: Dict mapping paragraph_idx -> relevance score (0-3)
        k: Cutoff for evaluation

    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    # DCG calculation
    dcg = 0.0
    for i, para_idx in enumerate(retrieved_paragraphs[:k]):
        relevance = relevance_judgments.get(para_idx, 0)
        if relevance > 0:
            # DCG formula: sum of (2^rel - 1) / log2(position + 1)
            dcg += (2**relevance - 1) / math.log2(i + 2)

    # IDCG calculation (perfect ranking - sort by relevance descending)
    ideal_relevances = sorted(relevance_judgments.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        if relevance > 0:
            idcg += (2**relevance - 1) / math.log2(i + 2)

    # nDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_mrr(
    retrieved_paragraphs: List[int], relevance_judgments: Dict[int, int]
) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.

    Args:
        retrieved_paragraphs: List of retrieved paragraph indices in ranked order
        relevance_judgments: Dict mapping paragraph_idx -> relevance score (0-3)

    Returns:
        Reciprocal rank (0.0 to 1.0) - position of first highly relevant (rel=3) result
    """
    for i, para_idx in enumerate(retrieved_paragraphs):
        # MRR: find first highly relevant result (relevance = 3)
        if relevance_judgments.get(para_idx, 0) == 3:
            return 1.0 / (i + 1)
    return 0.0


def run_test(vector_store: VectorStore, use_reranking: bool = False):
    """
    Run retrieval test on philosophy document paragraphs.

    Args:
        vector_store: Initialized vector store with philosophy document
        use_reranking: Whether to use cross-encoder reranking
    """
    method = "FAISS + Cross-Encoder Reranking" if use_reranking else "FAISS Only"
    print(f"\n{'=' * 80}")
    print(f"Testing: {method}")
    print(f"{'=' * 80}\n")

    embedding_service = EmbeddingService()
    reranker = CrossEncoderReranker() if use_reranking else None

    correct = 0
    total = len(TEST_QUERIES)

    # Track metrics overall
    ndcg_scores = []
    mrr_scores = []

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        relevance_judgments = test_case["relevance_judgments"]
        description = test_case["description"]

        # Get the highly relevant paragraph (relevance=3) for display
        highly_relevant = [p for p, rel in relevance_judgments.items() if rel == 3]
        expected_para = highly_relevant[0] if highly_relevant else None

        print(f"[{i}/{total}] {query}")
        if expected_para is not None:
            print(
                f"  Expected: Paragraph {expected_para} (+ {len(relevance_judgments)-1} other relevant)"
            )
        print(f"  Goal: {description}")

        # Search
        if use_reranking:
            # Embed query for reranking search
            query_embedding = embedding_service.embed_texts([query])[0]
            results, _ = vector_store.search_with_reranking(
                query_text=query,
                query_embedding=query_embedding,
                top_k_faiss=30,  # Retrieve all sentences to ensure all 50 paragraphs are candidates
                top_k_paragraphs=10,
                reranker=reranker,
                context_window=0,
            )
        else:
            # Embed query for FAISS search
            query_embedding = embedding_service.embed_texts([query])[0]
            results = vector_store.search(query_embedding=query_embedding, top_k=10)

        # Check if top result matches expected paragraph
        if results:
            # Extract paragraph indices from results
            retrieved_paragraphs = [r.paragraph_idx for r in results]

            # Calculate metrics with graded relevance
            ndcg = calculate_ndcg(retrieved_paragraphs, relevance_judgments, k=10)
            mrr = calculate_mrr(retrieved_paragraphs, relevance_judgments)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

            top_result = results[0]
            retrieved_para = top_result.paragraph_idx
            retrieved_relevance = relevance_judgments.get(retrieved_para, 0)

            # Check if top result is highly relevant
            is_correct = retrieved_relevance == 3
            status = "✓" if is_correct else ("~" if retrieved_relevance > 0 else "✗")

            relevance_labels = {
                3: "highly relevant",
                2: "relevant",
                1: "marginally relevant",
                0: "not relevant",
            }
            rel_label = relevance_labels.get(retrieved_relevance, "not relevant")

            print(f"  Retrieved: Paragraph {retrieved_para} ({rel_label}) {status}")
            print(f"  nDCG@10: {ndcg:.3f}, MRR: {mrr:.3f}")
            if retrieved_relevance < 3:
                print(f"    Top result: {top_result.paragraph_text[:100]}...")

            if is_correct:
                correct += 1
        else:
            print("  Retrieved: No results ✗")
            print(f"  nDCG@10: 0.000, MRR: 0.000")
            ndcg_scores.append(0.0)
            mrr_scores.append(0.0)

        print()

    # Summary
    accuracy = (correct / total) * 100
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    print(f"{'=' * 80}")
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"Average nDCG@10: {avg_ndcg:.3f}")
    print(f"Average MRR:     {avg_mrr:.3f}")
    print(f"{'=' * 80}\n")

    return {
        "accuracy": accuracy,
        "ndcg@10": avg_ndcg,
        "mrr": avg_mrr,
    }


def main():
    """Run the philosophy document retrieval test."""
    print("\n" + "=" * 80)
    print("PHILOSOPHY DOCUMENT RETRIEVAL TEST")
    print("Testing FAISS + Cross-Encoder nDCG on 30 similar utilitarianism paragraphs")
    print("=" * 80 + "\n")

    # Ingest document
    vector_store = ingest_document()

    # Test FAISS only
    faiss_results = run_test(vector_store, use_reranking=False)

    # Test FAISS + reranking
    reranking_results = run_test(vector_store, use_reranking=True)

    # Final comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"FAISS Only:")
    print(f"  Overall Accuracy:    {faiss_results['accuracy']:.1f}%")
    print(f"  Overall nDCG@10:     {faiss_results['ndcg@10']:.3f}")
    print(f"  Overall MRR:         {faiss_results['mrr']:.3f}")
    print()
    print(f"FAISS + Reranking:")
    print(f"  Overall Accuracy:    {reranking_results['accuracy']:.1f}%")
    print(f"  Overall nDCG@10:     {reranking_results['ndcg@10']:.3f}")
    print(f"  Overall MRR:         {reranking_results['mrr']:.3f}")
    print()

    # Calculate improvements (absolute and relative to FAISS baseline)
    print(f"Overall Improvement:")
    acc_diff = reranking_results["accuracy"] - faiss_results["accuracy"]

    # Relative improvement: (reranking - faiss) / faiss * 100
    ndcg_rel_imp = (
        (reranking_results["ndcg@10"] - faiss_results["ndcg@10"])
        / faiss_results["ndcg@10"]
        * 100
        if faiss_results["ndcg@10"] > 0
        else 0.0
    )
    mrr_rel_imp = (
        (reranking_results["mrr"] - faiss_results["mrr"]) / faiss_results["mrr"] * 100
        if faiss_results["mrr"] > 0
        else 0.0
    )
    print(f"  Accuracy:    {acc_diff:+.1f} pp")
    print(f"  nDCG@10:     {ndcg_rel_imp:+.1f}% relative improvement")
    print(f"  MRR:         {mrr_rel_imp:+.1f}% relative improvement")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
