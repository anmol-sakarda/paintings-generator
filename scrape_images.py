'''
Scraping images from MET database
'''

import urllib.request
import requests
import time
import random

# Using API to request
response = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects')
numOfArt = response.json().get('total')
count = 0

# iterating through all possible objects in the MET database.
for i in range(1, 450000):

    # randomly adding a timer every 50 requests to throw off the api to allow us to continue scraping
    if i % 50 == 0:
        sleeping = random.randint(2, 4)
        time.sleep(sleeping)

    # requesting from the MET API
    json = requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/' + str(i)).json()
    classification = json.get("classification")
    image_link = json.get("primaryImage")
    type_of_art = json.get("department")
    tags = json.get('tags')
    # checking to see if the classification i paintings and the type of art is not Asian art
    if (classification == "Paintings") and image_link != '' and type_of_art != 'Asian Art':
        remove = False
        if tags is not None:
            for tag in tags:
                term = tag.get('term')
                # Ensuring no portraits or photos of people.
                if term == 'Portraits' or term == 'Men' or term == 'Women' or term == 'Children' or term == 'Human Figures' or term == 'Profiles':
                    remove = True
                    break
            if remove is False:
                try:
                    urllib.request.urlretrieve(image_link, "scraped_images/image_" + str(i) + ".jpg")
                    print(i)
                    print(tags)
                    count += 1
                except:
                    pass

