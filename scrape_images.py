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
for i in range(1, 200000):

    if i % 50 == 0:
        sleeping = random.randint(2, 4)
        time.sleep(sleeping)

    json = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/' + str(i)).json()
    classification = json.get("classification")
    image_link = json.get("primaryImage")
    type_of_art = json.get("department")
    tags = json.get('tags')
    if (classification == "Paintings") and 'Men' not in tags and 'Women' not in tags and 'Children' not in tags and 'Portraits' not in tags and 'Human Figures' not in tags and image_link != '':
        try:
            urllib.request.urlretrieve(image_link, "Images/image_" + str(i) + ".jpg")
            print(image_link)
            count += 1
        except:
            pass
    if i % 1000 == 0:
        print(i)
