# NLP_homework

Here is the code of the nlp homework. 

We implement the fact extraction and verification pipeline with three steps, consisting of abstract retrieval, rationale selection, and fact verification, and we fine-tune pretrained language models for fact-checking tasks. We implement continuous training methods to train the pretrained language model to learn domain knowledge for few sample oriented fact verification tasks. The SCIFACT, a dataset of expert-written scientific fact verification, is utilized to evaluate the performance of our model. 

The results show that our model achieves 61% and 77% on precision from the abstract level and the rationale level respectively, which is 30% absolute improvement compared with the baseline, indicating that our model can provide high-quality and credible verification for few sample oriented fact verification tasks. 

Based on our scientific fact verification model, we design and implement a scientific fact verification system. The system provides a visual Web interface for real-time interaction with users, offering scientific fact verification functions through both text input and document upload methods. 