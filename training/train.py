import argparse
import yaml

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Running model:", config["model"])
    print("Dataset:", config["dataset"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)