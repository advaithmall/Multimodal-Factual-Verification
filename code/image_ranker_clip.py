from PIL import Image
import requests
import os
from transformers import CLIPProcessor, CLIPModel



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


images_dir='/scratch/raghavd0/test_wiki_pics'
pages=os.listdir(f'{images_dir}/')
# print(f'{pages} folders present in images directory')



for p in pages:
    pageName=p.replace('_',' ')
    print(f'Parsing {pageName}')


    images=os.listdir(f'{images_dir}/{p}/')
    imageList=[]
    for im_path in images:
        try:
            imageList.append(Image.open(f'{images_dir}/{p}/{im_path}'))
        except:
            continue

    if len(imageList) == 0:
        print(f'Nothing here')
        continue
    

    inputs = processor(text=[pageName], images=imageList, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    image_sim = outputs.logits_per_image.flatten().softmax(dim=0)  # this is the image-text similarity score
    
    im_sim_sorted=sorted(zip(image_sim,range(len(image_sim))),reverse=True)
    topk=2

    for im_sim in im_sim_sorted[:topk]:
        try:
            # rename file to filter0.jpg
            im_path=images[im_sim[1]]
            print(f'{im_path} made the cut')
            os.rename(f'{images_dir}/{p}/{im_path}',f'{images_dir}/{p}/filter{im_path}')
        except:
            pass
    
    print(f'\n\n')


