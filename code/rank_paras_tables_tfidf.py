import json
file1 = open("feverous_train.jsonl", "r")

data = []
for line in file1:
    data.append(json.loads(line))

from pprint import pprint
data = data[1:]
import regex as re

fin_data = {}
for item in data:
    claim = item['claim']
    label = item['label']
    evid = item['evidence'][0]['content'][0]
    evid = evid.split("_")[0]
    label_int = -1
    if label == "SUPPORTS":
        label_int = 1
    elif label == "REFUTES":
        label_int = 0
    elif label == "NOT ENOUGH INFO":
        label_int = 2
    fin_data[claim] = [evid, label_int]

import gensim  
embeddings = gensim.models.KeyedVectors.load_word2vec_format('/home/advaith/Desktop/ada_backup_sept/GoogleNews-vectors-negative300.bin', binary=True)
from tqdm import tqdm
import torch
# load stop words from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import torch
for key in tqdm(fin_data.keys(), total = len(fin_data.keys())):
    row = fin_data[key]
    claim = key
    claim = claim.lower()
    # remove punctuations
    claim = re.sub(r'[^\w\s]', '', claim)
    claim = set(claim.split())
    # remove stop words
    claim = [word for word in claim if word not in stop_words]
    # 300 dim 0 tensor
    claim_vec = torch.zeros(300)
    for word in claim:
        if word in embeddings:
            claim_vec += embeddings[word]
    claim_vec /= len(claim)
    row.append(claim_vec)
    row.append(claim)
    # evidence, label, claim vec, claim set
    fin_data[key] = row

    
import torch
fin_data = torch.load("fin_data.pt")from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
frdb =  FeverousDB("feverous_wikiv1.db")
def rank_sents(claim, list_sents, n):
    if list_sents == []:
        return []
    # rank all sents in list_sents against claim using tf-idf
    # return top n sents
    top_n = []
    claim = claim.lower()
    # remove punctuations
    claim = re.sub(r'[^\w\s]', '', claim)
    claim = set(claim.split())
    # remove stop words
    claim = [word for word in claim if word not in stop_words]
    claim = " ".join(claim)
    claim = claim.split()
    # get tf-idf scores
    scores = []
    for sent in list_sents:
        sent = sent.lower()
        # remove punctuations
        sent = re.sub(r'[^\w\s]', '', sent)
        sent = set(sent.split())
        # remove stop words
        sent = [word for word in sent if word not in stop_words]
        sent = " ".join(sent)
        sent = sent.split()
        score = 0
        for word in claim:
            if word in sent:
                score += 1
        scores.append(score)
    # sort scores and sents in descending order
    sorted_scores, sorted_sents = zip(*sorted(zip(scores, list_sents), reverse=True))
    # return top n sents
    return sorted_sents[:n]
def rank_tables(claim, list_tables, n):
    if list_tables == []:
        return []
    # remove spaces and non alphanumeric characters from claim
    claim = claim.lower()
    # remove punctuations
    claim = re.sub(r'[^\w\s]', '', claim)
    scores = []
    claim = set(claim.split())
    for table in list_tables:
        # remove spaces and non alphanumeric characters from table
        table = table.lower()
        # remove punctuations
        table = re.sub(r'[^\w\s]', '', table)
        table = set(table.split())
        # get tf-idf score
        score = 0
        for word in claim:
            if word in table:
                score += 1
        scores.append(score)
    # sort scores and tables in descending order
    sorted_scores, sorted_tables = zip(*sorted(zip(scores, list_tables), reverse=True))
    # return top n tables
    return sorted_tables[:n]
            
        
 import torch
from tqdm import tqdm
from feverous.utils.wiki_page import WikiPage
evidence = {}
for key in tqdm(fin_data.keys(), total = len(fin_data.keys())):
    #print(key)
    row = fin_data[key]
    evid = row[0]
    #print(evid)
    page_json = db.get_doc_json(evid)
    #print(evid)
    #pprint(page_json)
    wiki_page = WikiPage(evid, page_json)
    listt = wiki_page.page_order
    target = listt[-1]
    prev_elements = wiki_page.get_previous_k_elements(target, k=len(listt)-1)
    sent_list = []
    table_list = []
    for item in prev_elements:
        typee = str(type(item))
        if 'sentence' in typee:
            content = str(item)
            # remove non alpha numeric characters
            content = re.sub(r'[^A-Za-z0-9]+', ' ', content)
            # replace multiple spaces  tabs newlines with single space
            content = re.sub(r'\s+', ' ', content)
            sent_list.append(content)
        elif 'table' in typee:
            content = str(item)
            table_list.append(content)
    #pprint(sent_list)
    top_n = rank_sents(key, sent_list, 5)
    top_tables = rank_tables(key, table_list, 5)
    loc_dict = {}
    loc_dict['sent'] = top_n
    loc_dict['table'] = top_tables
    loc_dict['evid'] = evid
    evidence[key] = loc_dict
    torch.save(evidence, "evidence.pt")


        