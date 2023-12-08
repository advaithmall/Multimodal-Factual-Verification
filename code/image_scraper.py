import wikipedia
import re
import os
import requests
import torch

base="/scratch/raghavd0/"
try:
    os.mkdir(base)
except:
    pass

titles=torch.load('final_names.pt')
# print(len(titles))
# titles=titles[:10]
# exit(0)

# titles=['Narendra Modi']
for title in titles:

    print(f'----------------------------------------------------->Querying Wikipedia page {title}')
    try:
        page=wikipedia.page(title)
    except:
        continue

    # print(len(page.images))
    links=[]
    for img_url in page.images:
        if re.search("\.(jpg|png|jpeg|JPEG|JPG|PNG)$",img_url):
            links.append(img_url)
    
    print(f'{len(links)} Images found')


    page_dir=f'{base}{title.replace(" ","_")}/'
    try:
        os.mkdir(page_dir)
    except:
        pass

    for img_count,img_url in enumerate(links):
    
        # Send an HTTP GET request to the image URL
        print(f'Sending request to {img_url}')
        response = requests.get(img_url,headers={'User-Agent':'Python Wikipedia Scraper/1.0 (Contact: sahishnaadvaith@gmail.com)'})

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Get the content of the response (image data)
            image_data = response.content

            # Specify the file name to save the image
            file_name = f'{page_dir}{img_count}{img_url.strip()[-4:]}'

            # Save the image to a file
            with open(file_name, 'wb') as file:
                file.write(image_data)

            print(f"Image saved as {file_name}\n")
        else:
            print(f"Failed to download the image. Status code: {response.status_code}\n")
    
    print(f'\n\n')