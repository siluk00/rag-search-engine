#!/usr/bin/env python3

import argparse
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_movies():
    with open('data/movies.json', 'r') as f:
                return json.load(f)['movies']


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") # subcomnmands like git add

    #verify_parser parser
    subparsers.add_parser("verify", help="Verify parser") #name + search + query

    #embed_text parser
    embed_text_parser = subparsers.add_parser("embed_text", help="Put the text to be embedded") #name + search + query
    embed_text_parser.add_argument("term", type=str, help="Term to be embedded")

    #verify_embeddings parser
    subparsers.add_parser("verify_embeddings", help="Verify embeddings") 

    #embed_query parser
    embed_query = subparsers.add_parser("embedquery", help="Put the text to be embedded") 
    embed_query.add_argument("query", type=str, help="Query to be embedded")
 
    #chunk parser
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text") 
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="limit size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Amount of overlap in words")

    #semantic_chunk parser
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk text") 
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="limit size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="Amount of overlap in words")

    #embed_chunks parser
    subparsers.add_parser("embed_chunks", help="Embed all movies into chunks") 

    #search chunk query parser
    search_chunked = subparsers.add_parser("search_chunked", help="Search for the best similarities witth chunked data") 
    search_chunked.add_argument("query", type=str, help="Query to be searched")
    search_chunked.add_argument("--limit", type=int, nargs='?', default=5, help="limit search")

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
        case "chunk":
            words = args.text.split()
            n = args.chunk_size
            step = args.chunk_size - args.overlap
            chunks = [words[i:i+n] for i in range(0, len(words), step)]
            print(f"Chunking {len(args.text)} characters")
            i = 1
            for chunk in chunks:
                print(f"{i}. {" ".join(chunk)}")
                i+=1
        case "semantic_chunk":
            from lib.semantic_search import semantic_chunk
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            i = 1
            for chunk in chunks:
                print(f"{i}. {chunk}")
                i+=1

        case 'embed_chunks':
            from lib.semantic_search import ChunkedSemanticSearch
            movies_dict = load_movies()
            semantic_search = ChunkedSemanticSearch()
            embeddings = semantic_search.load_or_create_chunk_embeddings(movies_dict)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case 'search_chunked':
            from lib.semantic_search import ChunkedSemanticSearch
            movies_dict = load_movies()
            semantic_search = ChunkedSemanticSearch()
            semantic_search.load_or_create_chunk_embeddings(movies_dict)
            response = semantic_search.search_chunks(args.query, args.limit)
            for i in range(len(response)):
                print(f"\n{i+1}. {response[i]['title']} (score: {response[i]['score']:.4f})")
                print(f"   {response[i]['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()