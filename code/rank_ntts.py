from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
db =  FeverousDB("feverous_wikiv1.db")
import torch 

final_entities = torch.load("use_this.pt")


import regex as re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def get_summary(entity):
    page_json = db.get_doc_json(entity.title())
    if page_json is None:
        return ""
    wiki_page = WikiPage(entity, page_json)
    listt = wiki_page.page_order
    index = 0
    for i in range(len(listt)):
        if listt[i] == 'section_1':
            index = i
            break
    # if sections 1 not in list, use all elements
    if index == 0:
        index = len(listt)
        prev_elements = wiki_page.get_previous_k_elements(listt[-1], k= index) # Gets Wiki element before sentence_5
    else:
        prev_elements = wiki_page.get_previous_k_elements('section_1', k= index)
    summary = ""
    for item in prev_elements:
        #print(type(item))
        type_l = str(type(item))
        if 'sentence' in type_l or 'table' in type_l:
            content = str(item)
            # using regex replace non alpha numeric characters with space
            content = re.sub(r'[^A-Za-z0-9]+', ' ', content)
            # replace multiple spaces  tabs newlines with single space
            content = re.sub(r'\s+', ' ', content)
            summary += content + ' '
    return summary


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

ranked_nnts = {}
for key in final_entities.keys():
    keys = list(final_entities.keys())
    key_scores = {}
    # encode the claim
    claim = key
    encoded_claim = encode(claim)
    for key in keys:
        summary = get_summary(key)
        if summary == "":
            continue
        encoded_summ = encode(summary)
        score = torch.cosine_similarity(encoded_claim, encoded_summ, dim=0)
        key_scores[key] = score
    # sort the key_scores in descending order
    sorted_keys = sorted(key_scores, key=key_scores.get, reverse=True)
    # get the top  keys
    k = 3
    top_10 = sorted_keys[:3]
    ranked_nnts[claim] = top_10
    torch.cuda.empty_cache()
    


