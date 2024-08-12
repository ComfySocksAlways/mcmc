import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sampling and transformation with specified parameters.")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    
    args = parser.parse_args()
    
    print(args.n_samples, args.verbose)