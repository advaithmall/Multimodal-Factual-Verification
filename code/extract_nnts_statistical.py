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
        ngrams.append(ngram[:-1])
        ngrams.append(ngram[1:])

        ngrams.append(ngram)
    return ngrams

def get_all_ngrams(sentence):
    all_ngrams = []
    for n in range(1, max+1):  # Generate n-grams from length 1 to 6
        ngrams = generate_ngrams(sentence, n)
        all_ngrams.extend(ngrams)
    return all_ngrams


final_entities = {}
verification_scores = {}
from tqdm import tqdm 
k = 0
import torch
from pprint import pprint
import wikipedia
for claim in tqdm(claims, total = 1200):
    #print(org_claim)
    org_claim = claim
    no_punctuations = re.sub(r'[^\w\s\-]', ' ', claim)
    # Replace \n, \t, and multiple spaces with single spaces
    claim = re.sub(r'\s+', ' ', no_punctuations).strip()
    #claim = claim.title()
    result = get_all_ngrams(claim)
    entity_set = set(result)
    entity_set = entity_set - stop_words
    verified_entities = []
    for word in entity_set:
        word = word.strip()
        word = word.split()
        word = " ".join(word)
        if db.get_doc_json(word) is not None:
            verified_entities.append(word)
        else:
            if db.get_doc_json(word.title()) is not None:
                 verified_entities.append(word)

    truth = claim_test_dict[org_claim][0]
    if truth in verified_entities:
        verification_scores[org_claim] = 1
    else:
       # check if truth is substring of claim
        verification_scores[org_claim] = 0
        if truth in claim:
            print("missed substring")
            print(len(truth.split()))
            targ = len(truth.split())
            print("trutn is : ", truth)
            # in entities set, print ngrams of length targ
            for word in entity_set:
                if len(word.split()) == targ:
                    print("---", word)


    verified_entities = set(verified_entities)
    final_entities[org_claim] = verified_entities
    torch.save(final_entities, "final_entities1.pt")
    torch.save(verification_scores, "verification_scores1.pt")


torch.save(final_entities, "final_entities1.pt")
torch.save(verification_scores, "verification_scores1.pt")
    

    