import concurrent.futures
from tqdm import tqdm
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage
from nltk.corpus import stopwords
import json
import regex as re
import torch
import wikipedia

# Initialize your database and other data
db = FeverousDB("feverous_wikiv1.db")
stop_words = set(stopwords.words('english'))

# Load claims and claim_test_dict
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

# Function to process a single claim
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


def process_claim(claim):
    org_claim = claim
    no_punctuations = re.sub(r'[^\w\s\-]', ' ', claim)
    claim = re.sub(r'\s+', ' ', no_punctuations).strip()
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
        verification_scores[org_claim] = 0
        if truth in claim:
            print("missed substring")
            print(len(truth.split()))
            targ = len(truth.split())
            print("truth is: ", truth)
            for word in entity_set:
                if len(word.split()) == targ:
                    print("---", word)

    verified_entities = set(verified_entities)
    final_entities[org_claim] = verified_entities

# Create a ThreadPoolExecutor with 20 workers
max = 14
final_entities = {}
verification_scores = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    # Use tqdm to track progress
    for _ in tqdm(executor.map(process_claim, claims), total=len(claims)):
        pass

# Save results after processing all claims
torch.save(final_entities, "final_entities.pt")
torch.save(verification_scores, "verification_scores.pt")


