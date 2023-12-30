import jsonlines

dataset = '../data/claims_test.jsonl'
rationale = '../prediction/rationale_selection_test_scibert_mlm.jsonl'
fact_check = '../prediction/kgat_test_roberta_large_mlm.json_pred.jsonl'

# choose abstract
# doc_ids: dict of --cid: {"43385013": "SUPPORT", ...}
doc_ids = {}
for retrieval in jsonlines.open(fact_check):
    aid = {}
    for k, v in retrieval['labels'].items():
        if v['label'] == "SUPPORT":
            aid.update({k: 'SUPPORT'})
        if v['label'] == "CONTRADICT":
            aid.update({k: 'CONTRADICT'})
    doc_ids.update({retrieval["claim_id"]: aid})

# add evidence
# results: dict of {"id": 1, "evidence": {"43385013": {"sentences": [1, 2], "label": "SUPPORT"}, ...}}
results = []
for retrieval in jsonlines.open(rationale):
    id = retrieval["claim_id"]
    result = {"id": id, "evidence": {}}
    for k, v in retrieval["evidence"].items():
        if k in doc_ids[id].keys():
            result["evidence"].update({k: {"sentences": v, "label": doc_ids[id][k]}})
    results.append(result)

# save results
with jsonlines.open("./test_results.jsonl", 'w') as f:
    for r in results:
        f.write(r)