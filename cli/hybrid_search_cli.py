import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize values") #name + search + query
    normalize_parser.add_argument('values', nargs='*', type=float, help='One or more float values')

    args = parser.parse_args()

    match args.command:
        case 'normalize':
            from lib.hybrid_search import normalize
            normalized_list = normalize(args.values)
            for item in normalized_list:
                print(f"{item:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()