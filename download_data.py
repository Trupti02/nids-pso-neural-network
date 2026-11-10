import urllib.request
from config import DATA_DIR, TRAIN_FILE, TEST_FILE, DOWNLOAD_BASE

def main():
    DATA_DIR.mkdir(exist_ok=True)
    for path in (TRAIN_FILE, TEST_FILE):
        if path.exists():
            print("Already exists:", path.name)
            continue
        url = DOWNLOAD_BASE + path.name.replace("+", "%2B")
        print("Downloading", path.name)
        urllib.request.urlretrieve(url, path)
    print("Done.")

if __name__ == "__main__":
    main()