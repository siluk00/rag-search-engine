#!/usr/bin/env python3

import argparse, json, string
from nltk.stem import PorterStemmer



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = []

            # print the search query here
            with open("data/movies.json", 'r') as f:
                dict = json.load(f)

            for entry in dict["movies"]: 
                stopwords = []
                with open("data/stopwords.txt", 'r') as f:
                    txt = f.read()
                    stopwords = txt.splitlines()

                stemmer = PorterStemmer()

                table = str.maketrans("", "", string.punctuation)
                entry_tokens = entry["title"].lower().translate(table).split(" ")
                entry_tokens = list(filter(lambda x: x != "", entry_tokens))
                entry_tokens = list(filter(lambda x: x not in stopwords, entry_tokens))
                entry_tokens = list(map(lambda x: stemmer.stem(x), entry_tokens))
                query_tokens = args.query.lower().translate(table).split(" ")
                query_tokens = list(filter(lambda x: x != "", query_tokens))
                query_tokens = list(filter(lambda x: x not in stopwords, query_tokens))
                query_tokens = list(map(lambda x: stemmer.stem(x), query_tokens))

                if any(any(any(q in e for e in entry_tokens) for q in query_tokens)):
                    results.append(entry)

            print(f"Searching for: {args.query}")
            for i in range(len(results)):
                if i == 5:
                    break
                print(f"{i+1}. {results[i]["title"]}")
        case _:
            parser.print_help()

    

if __name__ == "__main__":
    main()