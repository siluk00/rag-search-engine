#!/usr/bin/env python3

import argparse, json, string
from nltk.stem import PorterStemmer


def tokenize_input(phrase, stopwords, table, stemmer):
    tokens = phrase.lower().translate(table).split(" ") #uncapitalize and remove punctuation to list
    tokens = list(filter(lambda x: x != "",tokens)) #removes blank
    tokens = list(filter(lambda x: x not in stopwords, tokens)) #removes words without meaning
    tokens = list(map(lambda x: stemmer.stem(x),   tokens)) #turns words to their stem
    return tokens    

def tokenize_text(dict, args): #a dictionary and a string args
    results = []
    
    stopwords = []
    with open("data/stopwords.txt", 'r') as f:
        txt = f.read()
        stopwords = txt.splitlines()
    
    stemmer = PorterStemmer()
    table = str.maketrans("", "", string.punctuation) #table of transformation, puntuation -> ""
    query_tokens = tokenize_input(args, stopwords, table, stemmer)

    for entry in dict["movies"]: 
        entry_tokens = tokenize_input(entry["title"], stopwords, table, stemmer)
        if any(any(q in e for e in entry_tokens) for q in query_tokens):
            results.append(entry)

    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands") # subcomnmands like git add

    search_parser = subparsers.add_parser("search", help="Search movies using BM25") #name + search + query
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            dict = {}
            with open("data/movies.json", 'r') as f:
                dict = json.load(f)

            results = tokenize_text(dict, args)

            print(f"Searching for: {args.query}")
            for i in range(len(results)):
                if i == 5:
                    break
                print(f"{i+1}. {results[i]["title"]}")
        case _:
            parser.print_help()

    

if __name__ == "__main__":
    main()