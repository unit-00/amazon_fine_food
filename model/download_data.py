import requests
from pathlib import Path

base_dir = Path(__file__).resolve(strict=True).parent

def download_data():
    """
    Download dataset from Stanford's Snap 
    and write to data directory
    """
    url = 'https://snap.stanford.edu/data/finefoods.txt.gz'
    response = requests.get(url)

    filename = url.split('/')[-1]

    with open(base_dir.joinpath('data', filename), 'wb') as f_in:
        f_in.write(response.content)
