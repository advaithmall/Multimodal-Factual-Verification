import json
import csv
import tqdm
import numpy as np
import nltk

import wikipediaapi
import wikipedia

user_agent = "Python Wikipedia Scraper/1.0 (Contact: sahishnaadvaith@gmail.com)"

# Create a Wikipedia API object
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent=user_agent
)


# Specify the title of the Wikipedia page you want to access
page_title = "modi"

# Get the Wikipedia page
page = wiki_wiki.page(page_title)


import requests
from bs4 import BeautifulSoup
url = page.fullurl
response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

# Find and extract image URLs
image_tags = soup.find_all('img')
image_urls = [img['src'] for img in image_tags]

# Print image URLs
file1 = open("images.txt", "w")
print("\nImage URLs from the Wikipedia page:")
links = []
for image_url in image_urls:
    print(image_url, file = file1)
    links.append(image_url)
print(links[10])

