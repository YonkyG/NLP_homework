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
# results: list of --{"claim_id": 1, "evidence": {"43385013": {"label": "SUPPORT", "sentences": [1, 2]}, ...}}
results = []
for retrieval in jsonlines.open(rationale):
    id = retrieval["claim_id"]
    result = {"claim_id": id, "evidence": {}}
    for k, v in retrieval["evidence"].items():
        if k in doc_ids[id].keys():
            result["evidence"].update({k: {"label": doc_ids[id][k], "sentences": v}})
    results.append(result)

tp = 0
pred = 0
gold = 0

for result in results:
    k = result["claim_id"]
    v = result["evidence"]
    data = dataset[k] # data: dict of line

    pred_doc_ids = set(v.keys())
    true_doc_ids = set(data['evidence'].keys())

    # check evidence
    # golden_evidence: dict of "5953485": [{"sentences": [1], "label": "SUPPORT"}, ...]
    golden_evidence = data['evidence']
    for abs_id, content in v.items():
        # content: "43385013": {"label": "SUPPORT", "evidence": [1, 2]}
        if abs_id in golden_evidence.keys():
            if content['label'] == golden_evidence[abs_id][0]['label']:
                gold_sen_ids = golden_evidence[abs_id] # list of dict
                for gold_sen_id_dict in gold_sen_ids:
                    if set(content["sentences"]).issuperset(set(gold_sen_id_dict['sentences'])):
                        tp += 1
                        break
            # else:
            #     print(content['label'], golden_evidence[abs_id][0]['label'])
    
    pred += len(pred_doc_ids)
    gold += len(true_doc_ids)

precision = tp / pred
recall = tp / gold
f1_score = 2 * precision * recall / (precision + recall)

print(f'TP: {round(tp, 4)}')
print(f'PRED: {round(pred, 4)}')
print(f'GOLD: {round(gold, 4)}')
print(f'Precision: {round(precision, 4)}')
print(f'Recall:    {round(recall, 4)}')
print(f'F1:        {round(f1_score, 4)}')