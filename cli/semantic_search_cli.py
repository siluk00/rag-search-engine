#!/usr/bin/env python3

import argparse
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") # subcomnmands like git add

    #verify_parser parser
    subparsers.add_parser("verify", help="Verify parser") #name + search + query

    #embed_text parser
    embed_text_parser = subparsers.add_parser("embed_text", help="Put the text to be embedded") #name + search + query
    embed_text_parser.add_argument("term", type=str, help="Term to be embedded")

    #verify_embeddings parser
    subparsers.add_parser("verify_embeddings", help="Verify embeddings") #name + search + query

    #embed_query parser
    embed_query = subparsers.add_parser("embedquery", help="Put the text to be embedded") #name + search + query
    embed_query.add_argument("query", type=str, help="Query to be embedded")

    #search query parser
    search_query = subparsers.add_parser("search", help="Search for the best similarities") #name + search + query
    search_query.add_argument("query", type=str, help="Query to be searched")
    search_query.add_argument("--limit", type=int, nargs='?', default=5, help="limit search")

    args = parser.parse_args()

    

    match args.command:
        case "verify":
            from lib.semantic_search import verify_model
            verify_model()
        case "embed_text":
            from lib.semantic_search import embed_text
            embed_text(args.term)
        case "verify_embeddings":    
            from lib.semantic_search import verify_embeddings
            verify_embeddings()
        case "embedquery":
            from lib.semantic_search import embed_query_text
            embed_query_text(args.query)
        case "search":
            from lib.semantic_search import SemanticSearch
            semantic_search = SemanticSearch()
            with open('data/movies.json', 'rb') as f:
                documents = json.load(f)
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            counter = 1
            for entry in results:
                print(f"{counter}. {entry["title"]} (score: {entry["score"]:.4f})")
                #print(f"   {entry["description"]}")
                counter += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()