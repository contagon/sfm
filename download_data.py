import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=1Wv04X0K8ja-EctX4ThN0G72Bd_z7xhWg"
zip = "data/moose.zip"
out = "data/"

gdown.download(url, zip, quiet=False)

print("\nUnzipping...")
with zipfile.ZipFile(zip, 'r') as zip_ref:
    zip_ref.extractall(out)

print("\nCleaning up...")
os.remove(zip)

print("\nDone!")
