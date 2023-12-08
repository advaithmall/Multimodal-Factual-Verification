from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
db =  FeverousDB("feverous_wikiv1.db")
import torch 
import regex as re
import operator
from transformers import BertTokenizer, BertModel

from tqdm import tqdm
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
final_entities = torch.load("use_this.pt")

claims = list(final_entities.keys())
rel_sent = {}
def encode(claim):
    if len(claim.split()) > 256:
        total_len = len(claim.split())
        claim = claim.split()
        # make fin_rerp an eempty tensor of size 256
        fin_rep = torch.zeros(768)
        # run through the claim in chunks of 256
        for i in range(0, total_len, 256):
            # get the chunk
            chunk = claim[i:i+256]
            # convert chunk to string
            chunk = " ".join(chunk)
            # encode the chunk
            tokenized_input = tokenizer.encode(chunk, return_tensors='pt').to(device)
            # get the embedding
            encoded_chunk = model(tokenized_input)
            # add the embedding to fin_rep
            fin_rep = fin_rep.to("cpu") + encoded_chunk[0][0][0].to("cpu")
            torch.cuda.empty_cache()

        # divide fin_rep by the number of chunks
        fin_rep = fin_rep/((total_len//256)+1)
        fin_rep = fin_rep
        return fin_rep
    else:
        tokenized_input = tokenizer.encode(claim, return_tensors='pt').to(device)
        encoded_claim = model(tokenized_input)
        encoded_claim = encoded_claim[0][0][0]
        torch.cuda.empty_cache()
        return encoded_claim

k = 1
for claim in tqdm(claims, total = len(claims)):
    print("--->", k)
    k+=1
    entity = final_entities[claim]
    page_json = db.get_doc_json(entity)
    wiki_page = WikiPage(entity,page_json)
    listt = wiki_page.page_order
    target = listt[-1]
    prev_elements = wiki_page.get_previous_k_elements(target, k=len(listt)-1)
    fin_list = []
    for item in prev_elements:
        typee = str(type(item))
        if 'sentence' in typee or 'table' in typee:
            content = str(item)
            # remove non alpha numeric characters
            content = re.sub(r'[^A-Za-z0-9]+', ' ', content)
            # replace multiple spaces  tabs newlines with single space
            content = re.sub(r'\s+', ' ', content)
            fin_list.append(content)
    encoded_claim = encode(claim)
    item_score = {}
    print(len(fin_list))
    for item in fin_list[:300]:
        #print(len(item.split()))
        item = item.split()
        item = " ".join(item[:200])
        encoded_item = encode(item)
        #encoded_item = encoded_item[0][0][0]
        score = torch.cosine_similarity(encoded_item.to("cpu"), encoded_claim.to("cpu"), dim=0)
        item_score[item] = score
        torch.cuda.empty_cache()

    sorted_x = sorted(item_score.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    fin_list = []
    for item in  sorted_x:
        fin_list.append(item[0])
    final = fin_list[:10]
    rel_sent[claim] = final

    torch.cuda.empty_cache()
torch.save(rel_sent, "rel_sent.pt")




            

    
           


