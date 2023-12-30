from flask import Flask, render_template, request, send_file
import os
import jsonlines

app = Flask(__name__)

@app.route('/upload_process')
def upload_process():
    text = request.args.get('text')
    processed_text = validate_text(text)
    return processed_text

@app.route('/process')
def process():
    text = request.args.get('text')
    processed_text = validate_text(text)
    return processed_text

# calculate jsonl
def validate_text(text):
    txt_lst = list(filter(None, text.split('\n')))
    with jsonlines.open('./data/my_claims.jsonl', mode='w') as f: # write claims
        cid = 0
        for txt in txt_lst:
            cid += 1
            claim_dict = {'id': cid, 'claim': txt}
            f.write(claim_dict)
    
    #infenrence
    infer()

    # read results
    with jsonlines.open('./prediction/my_kgat_roberta_large_mlm_pred.jsonl', mode='r') as f:
        labels_lst = []
        for line in f:
            labels_lst.append(line.get('labels'))

    # read evidence
    with jsonlines.open('./prediction/my_rationale_selection_scibert_mlm.jsonl', mode='r') as f:
        evidence_lst = []
        for line in f:
            evidence_lst.append(line.get('evidence'))
    
    processed_text = ''
    for tx, lb, ev in zip(txt_lst, labels_lst, evidence_lst):
        processed_text += 'Claim: ' + tx + '\nClaim Label: '
        processed_text += output(lb, ev)
        processed_text += '\n\n'
    
    return processed_text

# jsonl to paragraph
def output(labels, evidence):
    res = 'NOT_ENOUGH_INFO' # result

    lbs = [label.get('label') for key, label in labels.items()] # claim label
    if 'SUPPORT' in lbs:
        res = 'SUPPORT'
    if 'CONTRADICT' in lbs:
        res = 'CONTRADICT'
    if 'SUPPORT' in lbs and 'CONTRADICT' in lbs:
        res = 'NOT_ENOUGH_INFO'

    doc_ids = [int(key) for key in labels]
    with jsonlines.open('./data/corpus.jsonl', mode='r') as f:
        e_num = 0
        for doc in f:
            if doc['doc_id'] in doc_ids:
                e_num += 1
                did = str(doc['doc_id']) # doc_id
                res += '\nNo.' + str(e_num) + ' Verification: '
                res += labels.get(did).get('label')
                res += '\nDocument Title: '
                res += doc['title']
                res += '\nRelated Evidence:'
                cnt = -1 # index of evidence
                num = 0 # display sentence num
                for stc in doc['abstract']:
                    cnt += 1
                    if cnt in evidence.get(did):
                        num += 1
                        res += " [" + str(num) + "]: "
                        res += stc
                if num == 0:
                    res += " None"

    return res

def infer():
    # inference.sh
    print("----------------abstract_retrieval----------------")
    cmd = 'python abstract_retrieval/tfidf.py \
    --corpus data/corpus.jsonl \
    --dataset data/my_claims.jsonl \
    --k 100 \
    --min-gram 1 \
    --max-gram 2 \
    --output prediction/my_abstract_retrieval_top100.jsonl'
    os.system(cmd)

    print("----------------abstract_rerank----------------")
    cmd = 'python ./abstract_rerank/inference.py \
        -checkpoint ./model/abstract_scibert_mlm/pytorch_model.bin \
        -corpus ./data/corpus.jsonl \
        -abstract_retrieval ./prediction/my_abstract_retrieval_top100.jsonl \
        -dataset ./data/my_claims.jsonl \
        -outpath ./prediction/my_abstract_rerank_mlm.jsonl \
        -max_query_len 32 \
        -max_seq_len 256 \
        -batch_size 32'
    os.system(cmd)

    print("----------------rationale_selection----------------")
    cmd = 'python ./rationale_selection/transformer.py \
    --corpus ./data/corpus.jsonl \
    --dataset ./data/my_claims.jsonl \
    --abstract-retrieval ./prediction/my_abstract_rerank_mlm.jsonl \
    --model ./model/rationale_scibert_mlm/ \
    --output-flex ./prediction/my_rationale_selection_scibert_mlm.jsonl'
    os.system(cmd)

    print("----------------kgat----------------")
    cmd = 'python ./kgat/test.py --outdir ./prediction \
    --corpus ./data/corpus.jsonl \
    --evidence_retrieval ./prediction/my_rationale_selection_scibert_mlm.jsonl \
    --dataset ./data/my_claims.jsonl \
    --checkpoint ./model/kgat_roberta_large_mlm/model.best.pt \
    --pretrain ./mlm_models/roberta_large_mlm \
    --name my_kgat_roberta_large_mlm \
    --roberta \
    --bert_hidden_dim 1024'
    os.system(cmd)


if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0')

