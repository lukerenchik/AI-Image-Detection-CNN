import os
from PIL import Image

folder_path = 'dataset'
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        filepath = os.path.join(root, filename)
        try:
            with Image.open(filepath) as img:
                img.verify()  # This will raise an exception if the image is invalid
        except Exception as e:
            print(f"Corrupted image found and removed: {filepath}")
            os.remove(filepath)  # or move it somewhere else
