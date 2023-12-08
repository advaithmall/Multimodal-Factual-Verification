from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
db =  FeverousDB("feverous_wikiv1.db")

#Paspels use languages including German, and Romanish only and has recorded a total of 94.83% of German speakers in the 2000 census.#
page_json = db.get_doc_json("Barack Obama")
#print(page_json)
wiki_page = WikiPage("Barack Obama", page_json)
# print article summary
# print all attributes of the WikiPage object
#print(wiki_page.__dict__)

# import set of stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

print("done importing and setting up database")

import json
import regex as re
file1 = open("feverous_train.jsonl", "r")
claims = []
data = []
claim_test_dict = {}
for line in file1:
    data.append(json.loads(line))
for i in range(1, len(data)):
    claim = data[i]['claim']
    claims.append(data[i]['claim'])
    evidence = data[i]['evidence']
    evid = evidence[0]['content'][0]
    evid1 = evid
    evid = evid.split("_")[0]
    claim_test_dict[claim] = [evid, evid]
max = 0
for key in claim_test_dict:
    item = claim_test_dict[key][0]
    n = len(item.split())
    if n>max:
        max = n
max = 14
print("max is: ", max)
print("Ran through all claims...")
def generate_ngrams(sentence, n):
    words = sentence.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        if "-" in ngram:
            temp_grams = ngram.split("-")
            ngrams.extend(temp_grams)
            ngrams.append(ngram.replace("-", " "))
        
        for i in range(1, 5):
            ngrams.append(ngram[:-i])
            ngrams.append(ngram[i:])

        ngrams.append(ngram)
    return ngrams

def get_all_ngrams(sentence):
    all_ngrams = []
    for n in range(1, max+1):  # Generate n-grams from length 1 to 6
        ngrams = generate_ngrams(sentence, n)
        all_ngrams.extend(ngrams)
    return all_ngrams

from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english")
final_entities = {}
verification_scores = {}
from tqdm import tqdm 
k = 0
import torch
from pprint import pprint
import wikipedia
for claim in tqdm(claims, total = len(claims)):
    #print(org_claim)
    org_claim = claim
    no_punctuations = re.sub(r'[^\w\s\-]', ' ', claim)
    # Replace \n, \t, and multiple spaces with single spaces
    claim = re.sub(r'\s+', ' ', no_punctuations).strip()
    claim1 = claim
    claim = Sentence(claim)
    tagger.predict(claim)
    entity_set = set()
    test_set = set()
    for entity in claim.get_spans('ner'):
        #print(claim1)
        word = entity.text
        loc_claim = claim1.split(word)
        # remove "" from loc_claim
        loc_claim = [x for x in loc_claim if x]
        #print(word,"---->")
        test_set.add(word)
        #print(loc_claim, len(loc_claim))
        l_ws = word.split()
        for item in l_ws:
            if item not in stop_words:
                entity_set.add(item)
        if len(loc_claim)==1:
            sent1 = loc_claim[0]
            word1 = sent1.split()[-1]
            word2 = sent1.split()[1]
            w1 = word1 + " " + word
            w2 = word + " " + word2
            entity_set.add(word)
            entity_set.add(w1)
            entity_set.add(w2)
        if len(loc_claim)>1:
            #print(loc_claim)
            sent1 = loc_claim[0]
            sent2 = loc_claim[1]
            sent1 = sent1.split()
            sent2 = sent2.split()
            word1 = sent1[-1]
            word2 = sent2[0]
            w1 = word1 + " " + word
            w2 = word + " " + word2
            w3 = word1 + " " + word + " " + word2
            #print(word, w1, w2, w3)
            entity_set.add(word)
            entity_set.add(w1)
            entity_set.add(w2)
            entity_set.add(w3)
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    final_set = set()
    for item in entity_set:
        if db.get_doc_json(item) is not None:
            final_set.add(item)
        else:
            if db.get_doc_json(item.title()) is not None:
                 final_set.add(item)
    truth = claim_test_dict[org_claim][0]
    #final_set.add(truth)
    if truth in final_set:
        verification_scores[org_claim] = 1
        final_entities[org_claim] = final_set
        #print("pass")
    else:
        print("fail")
        print(truth)
        print(test_set)
        print(claim1)
    #print("------------------------------------------------------------")
    if k == 1000:
        break
    k+=1
import torch
torch.save(final_entities, "final_entities1.pt")
torch.save(verification_scores, "verification_scores1.pt")