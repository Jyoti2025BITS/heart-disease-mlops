from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

DEST = Path("data/raw/processed.cleveland.data")


def main():

    DEST.parent.mkdir(parents=True, exist_ok=True)

    if not DEST.exists():
        print(f"Downloading dataset from {URL}")

        urlretrieve(URL, DEST)

    print(f"Dataset saved to {DEST}")

    print(f"File size: {DEST.stat().st_size} bytes")


if __name__ == "__main__":
    main()
