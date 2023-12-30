import jsonlines

dataset = '../data/claims_dev.jsonl'
rationale = '../prediction/rationale_selection_dev_scibert_mlm.jsonl'
fact_check = '../prediction/kgat_dev_roberta_large_mlm.json_pred.jsonl'

dataset = {data['id']: data for data in jsonlines.open(dataset)}

# choose abstract
# doc_ids: dict of --cid: {"43385013": "SUPPORT", ...}
doc_ids = {}
for retrieval in jsonlines.open(fact_check):
    aid = {}
    for k, v in retrieval['labels'].items():
        if v['label'] != "NOT_ENOUGH_INFO":
            aid.update({k: v['label']})
    doc_ids.update({retrieval["claim_id"]: aid})

# add evidence
# results: dict of --1: {"claim_id": 1, "evidence": {"43385013": {"label": "SUPPORT", "sentences": [1, 2]}, ...}}
results = {}
for retrieval in jsonlines.open(rationale):
    id = retrieval["claim_id"]
    result = {"claim_id": id, "evidence": {}}
    for k, v in retrieval["evidence"].items():
        if k in doc_ids[id].keys():
            result["evidence"].update({k: {"label": doc_ids[id][k], "sentences": v}})
    results.update({id: result})

tp = 0
pred = 0
gold = 0


for cid, data in dataset.items():
    # cal pred
    # predict_evidence: dict of "43385013": {"label": "SUPPORT", "sentences": [1, 2]}
    predict_evidence = results[cid]["evidence"]
    for pred_did, pred_content in predict_evidence.items():
        pred += len(pred_content["sentences"])
    # cal gold and tp
    # golden_evidence: dict of "5953485": [{"sentences": [1], "label": "SUPPORT"}, ...]
    golden_evidence = data['evidence']
    for gold_did, gold_content in golden_evidence.items():
        for piece in gold_content:
            gold += len(piece["sentences"])
            if gold_did in predict_evidence.keys():
                if set(predict_evidence[gold_did]["sentences"]).issuperset(set(piece["sentences"])):
                    tp += len(piece["sentences"])


precision = tp / pred
recall = tp / gold
f1_score = 2 * precision * recall / (precision + recall)

print(f'TP: {round(tp, 4)}')
print(f'PRED: {round(pred, 4)}')
print(f'GOLD: {round(gold, 4)}')
print(f'Precision: {round(precision, 4)}')
print(f'Recall:    {round(recall, 4)}')
print(f'F1:        {round(f1_score, 4)}')