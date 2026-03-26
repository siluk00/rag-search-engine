import argparse, json
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5,help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    with open('data/movies.json', 'r') as f:
        doc = json.load(f)['movies']

    with open("data/golden_dataset.json", 'r') as f:
        golden_dataset = json.load(f)["test_cases"]
    
    hybrid_search = HybridSearch(doc)
    for data in golden_dataset:
        result = hybrid_search.rrf_search(data["query"], 60, limit)
        retrieved = [item['title'] for item in result]
        relevant = []
        relevant_docs = data['relevant_docs']
        for r in result:
            if r['title'] in relevant_docs:
                relevant.append(r['title'])
        print(f"- Query: {data['query']}")
        precision = len(relevant)/limit
        print(f"  - Precision@{limit}: {precision:.4f}")
        recall = len(relevant)/len(relevant_docs)
        print(f"  - Recall@{limit}: {recall:.4f}")
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant: {", ".join(relevant)}") 


if __name__ == "__main__":
    main()