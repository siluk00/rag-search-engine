#!/usr/bin/env python3

import argparse
import os
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()