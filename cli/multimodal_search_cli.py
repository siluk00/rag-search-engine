import argparse, json
from lib.multimodal_search import verify_image_embedding, MultimodalSearch

def load_movies():
    with open('data/movies.json', 'r') as f:
        return json.load(f)['movies']

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    embed_parser = subparsers.add_parser("verify_image_embedding", help="Perform image embedding from path")
    embed_parser.add_argument("path", type=str, help="Path of image")

    img_search_parser = subparsers.add_parser("image_search", help="Perform image search")
    img_search_parser.add_argument("path", type=str, help="Path of image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            print("hello")
            verify_image_embedding(args.path)
        case "image_search":
            docs = load_movies()
            multi_modal_search = MultimodalSearch(docs)
            results = multi_modal_search.search_with_image(args.path)
            i = 1
            for result in results:
                print(f"{i}. {result["title"]} (similarity: {result["similarity"]:.3f})")
                print(f"   {result["description"]}")
                i += 1
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()