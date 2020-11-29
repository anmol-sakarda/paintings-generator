'''
Scraping images from MET database
'''
import pandas as pd
import requests

from bs4 import BeautifulSoup
import urllib.request
import requests
import time
import numpy as np
import time
import random

# Using API to request
response = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects')
numOfArt = response.json().get('total')
count = 0

# 322491,322495
for i in range(5000, 10000):

    if i % 50 == 0:
        time.sleep(1)

    json = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/' + str(i)).json()
    classification = json.get("classification")
    image_link = json.get("primaryImage")
    type_of_art = json.get("department")
    if (classification == "Paintings") and type_of_art == 'Modern and Contemporary Art' and image_link != '':
        try:
            urllib.request.urlretrieve(image_link, "Images/image_" + str(i) + ".jpg")
            print(image_link)
            count += 1
        except:
            pass


