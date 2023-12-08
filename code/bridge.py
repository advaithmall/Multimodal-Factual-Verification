import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch import nn, optim
from transformers import BertTokenizer, BertModel
from transformers import BridgeTowerProcessor, BridgeTowerModel
import os
def get_images(topic):
    topic = topic.replace(' ', '_')
    # in dir topic, search for filenames with "filter"
    # return list of filenames
    path = os.path.join(topic)
    files = os.listdir(path)
    img_list = []
    for file in files:
        if "filter" in file:
            img_list.append(file)
    return img_list
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fin_data = torch.load("fin_data.pt")
key = list(fin_data.keys())[0]

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")


class fused_model(nn.Module):
    def __init__(self):
        super(fused_model, self).__init__()
        self.bridge_model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")
        self.fin_layer = nn.Linear(1536*2, 3)
    def forward(self, encoded, encoded2):
        encoded = encoded.to(device)
        encoded2 = encoded2.to(device)
        encoded = self.bridge_model(**encoded)
        encoded2 = self.bridge_model(**encoded2)
        encoded = encoded.pooler_output
        encoded2 = encoded2.pooler_output
        # concat encoded and encoded2
        encoded = torch.cat((encoded, encoded2), 1)
        encoded = self.fin_layer(encoded)
        return encoded

import regex as re
evid = torch.load("evidence_0.pt")
from pprint import pprint
tot = 0
cor_tot = 0
loss_list = []
acc_list = []
correct_list = []
pred_list = []
item_list = list(evid.keys())
model = fused_model()
model = model.to(device)
optimizer = optim.Adam(model.fin_layer.parameters(), lr=0.000001)
criterion = nn.CrossEntropyLoss()
print("Loaded model")
for epoch in range(20):
    batch = 0
    for key in item_list[:2000]:
        #pprint(evid[key])
        sents = evid[key]['sent']
        tables = evid[key]['table']
        claim = key

        # combine all sents into 1 sent
        sent_lens = []
        for sent in sents:
            length = len(sent.split())
            sent_lens.append(length)
        # combine all tables into 1 table
        fin_table = []
        table_lens = []
        for table in tables:
            # remove punctuation and multiple spaces, newlines and tabs
            table = re.sub(r'[^A-Za-z0-9]+', ' ', table)
            table = re.sub(r'\s+', ' ', table)
            table = table
            fin_table.append(table)
            table_lens.append(len(table.split()))
        # combine fin_sent and fin_table
        # extends sents to fin_table
        fin = []
        fin.extend(sents)
        fin.extend(fin_table)
        # combine fin into 1 huge string
        print(len(fin))
        fin = fin[:5]
        fin = " ".join(fin)
        images = get_images(claim)
        image1 = images[0]
        image2 = images[1]
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        # encode image and text 
        encoded = processor(image1, fin, return_tensors="pt")
        encoded2 = processor(image2, fin, return_tensors="pt")
        # forward pass
        #print(encoded.keys())
        out = model(encoded, encoded2)
        item = fin_data[key]
        label = int(item[1])
        out = out.reshape(1,3)
        loss = criterion(out, torch.tensor([label]).to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if answer is correct, increment cor_tot
        if torch.argmax(out) == label:
            cor_tot+=1
        tot+=1
        pred_list.append(torch.argmax(out).item())
        correct_list.append(label)
        accu = cor_tot/tot
        loss_list.append(loss.item())
        acc_list.append(accu)
        print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", loss.item(), "Accuracy: ", accu)
        batch+=1
    
    
