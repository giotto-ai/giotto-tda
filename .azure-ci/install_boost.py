#!/usr/bin/env python

import os
from pathlib import Path
import urllib.request
import shutil
import zipfile


url = "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.zip"
boost_folder = r"C:\local"

Path(boost_folder).mkdir(parents=True, exist_ok=True)
zip_file = os.path.join(boost_folder, "1_72_0.zip")

with urllib.request.urlopen(url) as response, \
        open(zip_file, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(boost_folder)

os.remove(zip_file)
