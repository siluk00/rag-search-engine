import argparse, json, os
from lib.hybrid_search import HybridSearch


def rag_command(query, command, limit=10):
    with open('data/movies.json', 'r') as f:
        movies = json.load(f)['movies']
    
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=10)
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    print("Search results:")
    results_list = []

    for result in results:
        results_list.append(f"Title: {result["title"]}, Deescription: {result["document"]}")
        print(f" - {result['title']}")

    docs = "\n".join(results_list)
    if command=='rag':
        response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                    f"""You are a RAG agent for Hoopla, a movie streaming service.
                    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                    Provide a comprehensive answer that addresses the user's query.

                    Query: {query}

                    Documents:
                    {docs}

                    Answer:""")
        print("RAG Response:")
    elif command == 'sum':
        response = client.models.generate_content(model='gemma-3-27b-it', contents=\
                            f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Search results:
                {docs}

                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:""")
        print("LLM summary:")
    print(f"{response.text}")                              

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Perform mult-document summarization")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("limit", type=int, nargs='?', help="limit results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query, 'rag')
        case "summarize":
            query = args.query
            limit = args.limit
            rag_command(query, 'sum', limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()