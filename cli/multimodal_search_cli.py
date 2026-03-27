import argparse
from lib.multimodal_search import verify_image_embedding

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    embed_parser = subparsers.add_parser("verify_image_embedding", help="Perform image embedding from path")
    embed_parser.add_argument("path", type=str, help="Path of image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            print("hello")
            verify_image_embedding(args.path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()