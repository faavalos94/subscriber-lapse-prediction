"""Download and unzip MovieLens 25M into data/.

Usage: python src/download_data.py
"""
import urllib.request, zipfile, pathlib

URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA = pathlib.Path(__file__).parent.parent / "data"

def main():
    DATA.mkdir(exist_ok=True)
    zip_path = DATA / "ml-25m.zip"

    if not zip_path.exists():
        print(f"Downloading {URL} (~250MB)...")
        urllib.request.urlretrieve(URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)
    print(f"Done! Ratings at {DATA / 'ml-25m' / 'ratings.csv'}")

if __name__ == "__main__":
    main()
