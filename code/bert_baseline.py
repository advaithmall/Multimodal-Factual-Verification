import torch

from torch import nn, optim
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
fin_data = torch.load("fin_data.pt")
key = list(fin_data.keys())[0]
from pprint import pprint
print(key)
pprint(fin_data[key])
class fact_model(nn.Module):
    def __init__(self):
        super(fact_model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(2*768, 3)
        #self.softmax = nn.Softmax(dim=1)
    def forward(self, x, claim, x_lens, claim_len):
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
        #print(claim.shape, x.shape)
        x = torch.cat((x, claim), 0)
        x = self.linear(x)
        #x = self.softmax(x)
        return x
model = fact_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("anlp_proj_model2.pt")
model = model.to(device)
optimizer = optim.Adam(model.linear.parameters(), lr=0.000001)
criterion = nn.CrossEntropyLoss()
print("Loaded model")
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
print(len(item_list), "----------_>")
for epoch in range(50):
    batch = 0
    for key in item_list:
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
        if len(sents) <=0:
            continue
        claim = tokenizer(claim, padding=True, return_tensors='pt')['input_ids']
        sents = tokenizer(fin, padding=True, return_tensors='pt')['input_ids']
        sents.to(device)
        claim.to(device)
        out = model(sents, claim, fin_lens, claim_len)
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
    torch.save(model, "anlp_proj_model3.pt")
    torch.save(loss_list, "proj_loss_list3.pt")
    torch.save(acc_list, "proj_acc_list3.pt")
    torch.save(pred_list, "proj_pred_list3.pt")
    
        
