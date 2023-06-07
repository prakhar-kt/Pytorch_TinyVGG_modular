import os
import requests
import zipfile
from pathlib import Path

def get_data():

    data_path = Path('data')

    image_path = data_path / 'pizza_steak_sushi'

    if image_path.is_dir():
        print("Image dir already exists")

    else:
        print("Creating image dir...")
        image_path.mkdir(exist_ok=True, parents=True)
        

        url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

        request = requests.get(url)

        with open(image_path / 'pizza_steak_sushi.zip', 'wb') as f:
            f.write(request.content)

        with zipfile.ZipFile(image_path / 'pizza_steak_sushi.zip', "r") as zip_ref:
            print("Extracting files...")
            zip_ref.extractall(image_path)

        os.remove(image_path / "pizza_steak_sushi.zip")






