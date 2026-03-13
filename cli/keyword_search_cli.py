#!/usr/bin/env python3

import argparse, math
from inverted_index import InvertedIndex
from keyword_search import tokenize_input, tokenize_word
from constants import BM25_K1, BM25_B

def bm25_idf_command(term: str, inverted_index: InvertedIndex) -> float:
    load(inverted_index)
    return inverted_index.get_bm25_idf(tokenize_word(term))

def bm_tf_command(doc_id: int, term: str, inverted_index: InvertedIndex, k1=BM25_K1, b=BM25_B) -> float:
    load(inverted_index)
    return inverted_index.get_bm25_tf(doc_id, tokenize_word(term), k1, b)

    
def load(inverted_index):
    try:
        inverted_index.load()
    except Exception as e:
        print(f"cannot find file: {e}")
        exit(1)
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") # subcomnmands like git add

    #search parser
    search_parser = subparsers.add_parser("search", help="Search movies using BM25") #name + search + query
    search_parser.add_argument("query", type=str, help="Search query")

    #build parser
    subparsers.add_parser("build", help="Builds the index")

    #tf parser
    tf_parser = subparsers.add_parser("tf", help="Gets frequency of term in doc_id")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", help="Term to see frequency")

    #idf parser
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequencies")
    idf_parser.add_argument("term", help="Term to find inverse frequency")

    #tfidf parser
    tfidf_parser = subparsers.add_parser("tfidf", help="Gets tfidf for term in doccument with specified doc_id")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", help="Term to calculate tf-idf")

    #bm25_idf parser
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    #bm25_tf parser
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    #bm25search parser
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="limit search results")

    args = parser.parse_args()
    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            load(inverted_index)

            results = []
            input = tokenize_input(args.query)
            results = []
            for element in input:
                results.extend(inverted_index.get_document(element))


            print(f"Searching for: {args.query}")
            for i in range(len(results)):
                if i == 5:
                    break
                print(f"{i+1}. {inverted_index.docmap[results[i]]['title']}") 
        case "build":
            inverted_index.build()
            inverted_index.save()
        case "tf":
            load(inverted_index)
            token = tokenize_word(args.term)
            frequency = inverted_index.get_tf(args.doc_id, token)
            print(f"Term frequency of term {args.term} with doc_id {args.doc_id} is {frequency}")
        case "idf":
            load(inverted_index)
            token = tokenize_word(args.term)   
            ids = inverted_index.get_document(token)
            document_size = len(inverted_index.docmap)
            idf = math.log((document_size+1)/(len(ids)+1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            load(inverted_index)
            token = tokenize_word(args.term)
            document_size = len(inverted_index.docmap)
            ids = inverted_index.get_document(token)
            tfidf = inverted_index.get_tf(args.doc_id, token) * math.log((document_size+1)/(len(ids)+1))
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term, inverted_index)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm_tf_command(args.doc_id, args.term, inverted_index, args.k1, args.b)          # uses function default
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            load(inverted_index)
            inverted_index.bm25_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()