# Interim Submission
## Team 25: Advaith, Raghav, Balaji

### Link to Dataset and 60GB wiki data dump: https://github.com/Raldir/FEVEROUS

### Download all the datasets, models, the latest reportand the .pt files into the current directory from this link:https://iiitaphyd-my.sharepoint.com/:f:/g/personal/advaith_malladi_research_iiit_ac_in/EgbzpJJ21EhJpzf7YjxrcYgBXW_4kCI4yWwXX1wdwz655w?e=n6wEbn


### For statictical entity extraction:
```
python3 extract_ntts_statistical.py
```

### Neural Entity Extraction using Named Entity Recognition (extract_ntt_neural.py)

```
python3 extract_ntt_neural.py
```

### Verification of Entities and their presence in WikiData Dump (verify_ntts.py)
```
python3 verify_ntts.py
```

### Ranking of Entities based on the semantic similarity between the introductory articles in the Wiki page of the entity and the Claim, choose top k entities (rank_ntts.py)
```
python3 rank_ntts.py
```

### Image Extraction of all entities which was not present in the provided dataset and had to be implemented from scratch (extract_images.py)
```
python3 extract_images.py or python3 image_scraper.py
```

### Ranking of all paragraphs and tables of all k entities against the claim, selecting top i paragraphs and tables (bert_rank_paras.py) using bert encodings
```
python3 bert_rank_paras.py
```

### Ranking all paragraphs and tables to choose the top 5 tables and top 5 parahraphs using tf-idf 
```
python3 rank_paras_tables_tfidf.py

```

### To choose the top 2 images per claim using the CLIP model
```
python3 image_ranker_clip.py
```

### Run the implementation of the paper using just paras and tables:
```
python3 bert_baseline.py
```

### Run the Dino+Bert implementation:
```
python3 dino_bert.py
```

### Run the Bridge-Tower model:
```
python3 bridge.py
```
