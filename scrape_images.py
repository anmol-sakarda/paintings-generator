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
for i in range(78000, 200000):

    if i % 50 == 0:
        sleeping = random.randint(2, 4)
        time.sleep(sleeping)

    json = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/' + str(i)).json()
    classification = json.get("classification")
    image_link = json.get("primaryImage")
    type_of_art = json.get("department")
    tags = json.get('tags')
    if (classification == "Paintings") and image_link != '' and type_of_art != 'Asian Art':
        remove = False
        if tags is not None:
            for tag in tags:
                term = tag.get('term')
                if term == 'Portraits' or term == 'Men' or term == 'Women' or term == 'Children' or term == 'Human Figures' or term == 'Profiles':
                    remove = True
                    break
            if remove is False:
                try:
                    urllib.request.urlretrieve(image_link, "Images/image_" + str(i) + ".jpg")
                    print(i)
                    print(tags)
                    count += 1
                except:
                    pass

    if i % 1000 == 0:
        print(i)
