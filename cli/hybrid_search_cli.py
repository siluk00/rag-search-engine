import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    #normalize parser
    normalize_parser = subparsers.add_parser("normalize", help="Normalize values") #name + search + query
    normalize_parser.add_argument('values', nargs='*', type=float, help='One or more float values')
    
    #search query parser
    weighted_search_query = subparsers.add_parser("weighted-search", help="Search for the best similarities with weighted data") 
    weighted_search_query.add_argument("query", type=str, help="Query to be searched")
    weighted_search_query.add_argument("--alpha", type=float, nargs='?', default=0.5, help="weigth")
    weighted_search_query.add_argument("--limit", type=int, nargs='?', default=5, help="limit search")
    
    #rrf search query parser
    rrf_search_query = subparsers.add_parser("rrf-search", help="Search for the best similarities with ranked data") 
    rrf_search_query.add_argument("query", type=str, help="Query to be searched")
    rrf_search_query.add_argument("--k", type=int, nargs='?', default=60, help="k parameter")
    rrf_search_query.add_argument("--limit", type=int, nargs='?', default=5, help="limit search")

    args = parser.parse_args()

    match args.command:
        case 'normalize':
            from lib.hybrid_search import normalize
            normalized_list = normalize(args.values)
            for item in normalized_list:
                print(f"{item:.4f}")
        case 'weighted-search':
            from lib.hybrid_search import HybridSearch
            from semantic_search_cli import load_movies
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            i = 1
            for result in results:
                print(f"{i}. {result['title']}")
                print(f"Hybrid_score: {result['score']:.4f}")
                print(f"BM25: {result['bm_25_score']:.4f}, Semantic: {result['semantic_score']:.4f}")
                print(f"{result['document']}")
                i+=1
        case 'rrf-search':
            from lib.hybrid_search import HybridSearch
            from semantic_search_cli import load_movies
            hybrid_search = HybridSearch(load_movies())
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            i = 1
            for result in results:
                print(f"{i}. {result['title']}")
                print(f"RRF_score: {result['rrf_score']:.4f}")
                print(f"BM25 rank: {result['bm_25_score']}, Semantic rank: {result['semantic_score']}")
                print(f"{result['document']}")
                i+=1
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()