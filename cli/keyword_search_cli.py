#!/usr/bin/env python3

import argparse, math
from invertedIndex import InvertedIndex
from keyword_search import tokenize_input, tokenize_word

def load(inverted_index):
    try:
        inverted_index.load()
    except Exception as e:
        print("cannot find file")
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
    tf_parser.add_argument("doc_id", type=int, help="Id")
    tf_parser.add_argument("term", help="Term to see frequency")

    #idf parser
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequencies")
    idf_parser.add_argument("term", help="Term to find inverse frequency")

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
            #print(inverted_index.index)
            docs = inverted_index.get_document("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case "tf":
            load(inverted_index)
            token = tokenize_word(args.term)
            frequency = inverted_index.get_tf(args.doc_id, token)
            print(f"Term frequency of term {args.term} with doc_id {args.doc_id} is {frequency}")
        case "idf":
            load(inverted_index)
            tokens = tokenize_input(args.term)
            if len(tokens) == 0:
                print("invalid token")
                exit(1)
            elif len(tokens) == 1:
                token = tokens[0]    
            ids = inverted_index.get_document(token)
            document_size = len(inverted_index.docmap)
            idf = math.log((document_size+1)/(len(ids)+1))
            print(f"total docs: {document_size}")
            print(f"docs with term: {len(ids)}")
            print(f"index size: {len(inverted_index.index)}")
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()