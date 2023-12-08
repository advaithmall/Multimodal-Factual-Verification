import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch import nn, optim
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fin_data = torch.load("fin_data.pt")
key = list(fin_data.keys())[0]
from pprint import pprint
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
    

image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

class fact_model(nn.Module):
    def __init__(self):
        super(fact_model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.image_model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.image_layer = nn.Linear(768*3, 600)
        self.text_layer = nn.Linear(768*2, 400)
        self.judgement_layer = nn.Linear(1000, 3)
        self.softmax = nn.Softmax()
    def forward(self, x, claim, x_lens, claim_len, image1, image2):
        image1 = image1.to(device)
        image2 = image2.to(device)
        x = x.to(device)
        #rint(x.shape)
        claim = claim.to(device)
        x = x[:, :512]
        x = self.bert(x)
        # for each item in x, choose the x_lens[i]th item
        x_outs = []
        x = x[0]
        #print("===>", x.shape)
        for i in range(len(x)):
            x_outs.append(x[i][-1])
        x = torch.stack(x_outs)
        #print("=========>", x.shape)
        # add all items in x (10, 768) should become (1, 768)
        x = torch.sum(x, dim=0)
        claim = self.bert(claim)
        # select the claim_len[i]th item from each claim
        #print(claim[0].shape, claim_len)
        claim = claim[0]
        claim = claim[0][claim_len]
        txt_out =torch.cat((x, claim), 0)
        text_out = self.text_layer(txt_out)
        # now encode image 1 and image 2
        out1 = self.image_model(**image1)
        out1 = out1.last_hidden_state
        out2 = self.image_model(**image2)
        out2 = out2.last_hidden_state
        # out1 and out2 are of shapes torch.Size([1, 257, 768]), convert to (1, 768) by addign along dim=1
        out1 = torch.sum(out1, dim=1)
        out2 = torch.sum(out2, dim=1)
        # now concatenate out1, out2 and claim along dim = 0
        # reshape claim to 1*768
        claim = claim.reshape(1, 768)
        #print(out1.shape, out2.shape, claim.shape, "------------------------_>")
        out = torch.cat((out1, out2, claim), dim=1)
        # now pass out through the image layer
        out = self.image_layer(out)
        # now txt out and out are of shape (1, 400) and (1, 600), make 1,1000
        # reshape out to 600 dim
        out = out.reshape(600)
        #print(text_out.shape, out.shape, "------------------------_>")
        out = torch.cat((text_out, out), dim=0)
        out = self.judgement_layer(out)
        out = self.softmax(out)
        return out




model = fact_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam([
    {'params': model.image_layer.parameters()},
    {'params': model.text_layer.parameters()},
    {'params': model.judgement_layer.parameters()}
], lr=0.01)
criterion = nn.NLLLoss()

import torch
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
        fin_lens = []
        fin_lens.extend(sent_lens)
        fin_lens.extend(table_lens)
        claim_len = len(claim.split())
        claim = tokenizer(claim, padding=True, return_tensors='pt')['input_ids']
        sents = tokenizer(fin, padding=True, return_tensors='pt')['input_ids']
        images = get_images(key)
        image1 = images[0]
        image2 = images[1]
        image1 = Image.open(image1)
        image2 = Image.open(image2)
        sents.to(device)
        claim.to(device)
        image1 = image_processor(image1, return_tensors="pt")
        image2 = image_processor(image2, return_tensors="pt")
        out = model(sents, claim, fin_lens, claim_len, image1, image2)
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
    
    