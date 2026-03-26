import argparse
import os, json

def enhance_query(query, method):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    match method:
        case 'spell':
            response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                                              f"""Fix any spelling errors in the user-provided movie search query below.
                            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                            Preserve punctuation and capitalization unless a change is required for a typo fix.
                            If there are no spelling errors, or if you're unsure, output the original query unchanged.
                            Output only the final query text, nothing else.
                            User query: "{query}"
                            """
                                                         )
            new_query = response.text
        case 'rewrite':
            response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                                                      f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep the rewritten query concise (under 10 words)
                            - It should be a Google-style search query, specific enough to yield relevant results
                            - Don't use boolean logic

                            Examples:
                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            If you cannot improve the query, output the original unchanged.
                            Output only the rewritten query text, nothing else.

                            User query: "{query}"
                            """
                                                        )
            new_query = response.text
        case 'expand':
            response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                                                      f"""Expand the user-provided movie search 
                                                      query below with a few related terms that are likely to appear in movie plot summaries.
                                Keep expansions short and focused.
                                Add only 3 to 5 terms.
                                Prefer character, profession, setting, or theme words over abstract subject words.
                                Output only the additional terms; they will be appended to the original query.

                                Examples:
                                - "math movie" -> "mathematician iq intelligent academic"
                                - "science movie" -> "scientist laboratory professor discovery"
                                - "comedy with bear" -> "funny humor lighthearted"

                                User query: "{query}"
                                """)
            new_query = f"{query} {response.text}"
        case _:
            return ""
    
    print(f"Enhanced query ({method}): '{query}' -> '{new_query}'\n")
    return new_query

def evaluate(query, formatted_results):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                        f"""Rate how relevant each result is to this query on a 0-3 scale:

                        Query: "{query}"

                        Results:
                        {chr(10).join(formatted_results)}

                        Scale:
                        - 3: Highly relevant
                        - 2: Relevant
                        - 1: Marginally relevant
                        - 0: Not relevant

                        Do NOT give any numbers other than 0, 1, 2, or 3.

                        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                        [2, 0, 3, 2, 0, 1]"""
    )
    return json.loads(response.text)



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
    rrf_search_query.add_argument("--enhance", type=str, nargs='?', choices=['spell', 'rewrite', 'expand'], help='Query enhancement method')
    rrf_search_query.add_argument("--rerank-method", type=str, nargs='?', choices=['individual', 'batch', 'cross_encoder'], help='Query reranking method')
    rrf_search_query.add_argument("--evaluate", action='store_true', help='Evaluate results')

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
            from lib.rerank import individual_rerank, batch_rerank, cross_encoder_rerank
            from semantic_search_cli import load_movies
            hybrid_search = HybridSearch(load_movies())
            query = args.query

            if args.enhance is not None:
                query = enhance_query(query, args.enhance)

            results = hybrid_search.rrf_search(query, args.k,  5 * args.limit)

            if args.rerank_method is not None:
                if args.rerank_method == 'individual':
                    for result in results:
                        result["rerank"]=individual_rerank(query, result, args.rerank_method)  
                    results = sorted(results, key=lambda x: x["rerank"], reverse=True)
                elif args.rerank_method == 'batch':
                    movie_lines = []
                    for result in results:
                        movie_lines.append(f"id: {result['id']}, title: {result['title']}, description: {result['document']}")
                    str_to_proccess = "\n".join(movie_lines)
                    ranks = batch_rerank(query, str_to_proccess)
                    rank_map= {doc_id: rank for rank, doc_id in enumerate(ranks)}
                    results.sort(key=lambda x:rank_map.get(x['id'], len(results)))
                elif args.rerank_method == 'cross_encoder':
                    results = cross_encoder_rerank(query, results)


            results = results[:args.limit]
            evaluation = None

            if args.evaluate:
                movie_lines = []
                for result in results:
                    movie_lines.append(f"id: {result['id']}, title: {result['title']}, description: {result['document']}")
                str_to_proccess = "\n".join(movie_lines)
                evaluation = evaluate(query, str_to_proccess)

            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']}")
                if args.rerank_method == 'cross_encoder':
                    print(f"Cross Encoder Score: {result['new_score']}")
                print(f"RRF_score: {result['rrf_score']:.4f}")
                print(f"BM25 rank: {result['bm_25_score']}, Semantic rank: {result['semantic_score']}")
                if args.evaluate:
                    print(f"Evaluation: {evaluation[i]}/3")
                print(f"{result['document']}")
    

            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()