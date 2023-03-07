#!/usr/bin/env python
# coding: utf-8

# © 2020 Nokia
# 
# Licensed under the BSD 3 Clause license
# 
# SPDX-License-Identifier: BSD-3-Clause

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''
from codesearch.ensemble.ensemble_embedder import EnsembleEmbedder
from codesearch.utils import load_model
from codesearch.embedding_retrieval import EmbeddingRetrievalModel
from codesearch.evaluation import evaluate_and_dump
from codesearch.data import load_snippet_collection, EVAL_DATASETS, eval_datasets_from_regex
from codesearch.utils import Saveable


# In[ ]:


snippets_collection = "so-ds-feb20"
valid_dataset = "so-ds-feb20-valid"
test_dataset = "so-ds-feb20-test"
ncs_model = "./nbs/ncs/ncs_pls_work/"

# snippets_collection = "conala-curated"
# valid_dataset = "conala-curated-0.5-test"
# test_dataset = "conala-curated-0.5-test"
# ncs_model = "../ncs/conala/best_ncs_embedder/"

# snippets_collection = "staqc-py-cleaned"
# valid_dataset = "staqc-py-raw-valid"
# test_dataset = "staqc-py-raw-test"
# ncs_model = "../ncs/staqc-py/best_ncs_embedder/"


if valid_dataset and valid_dataset not in EVAL_DATASETS:
    raise ValueError()
test_datasets = eval_datasets_from_regex(test_dataset)
snippets = load_snippet_collection(snippets_collection)


# In[ ]:


model_names = [ncs_model, 
               "./pacsv1/use5-act=linear_sigmoid-sim=cosine-negsamples=5-lr=0.0001-dropout=0.0-date=31/use_steps=5100",]

    
#for weights in [[.4, 1.], [.5, 1.], [.6, 1.], [.7, 1.]]:

#    embedder = EnsembleEmbedder(model_names, weights)
#    retrieval_model = EmbeddingRetrievalModel(embedder)
#    retrieval_model.add_snippets(snippets)
#    config = {"weights": weights}
#    results = evaluate_and_dump(retrieval_model, config, ".", valid_dataset, test_datasets)
#    print(results)


# In[ ]:


#for weights in [[.3, 1.]]:
#    embedder = EnsembleEmbedder(model_names, weights)
#    retrieval_model = EmbeddingRetrievalModel(embedder)
#    retrieval_model.add_snippets(snippets)
#    config = {"weights": weights}
#    results = evaluate_and_dump(retrieval_model, config, "ensemble_evaluation_results", valid_dataset, test_datasets)
#    print(results)
from codesearch.data import load_jsonl

#print(snippets)
weights = [.3, 1.]
embedder = EnsembleEmbedder(model_names, weights)
retrieval_model = EmbeddingRetrievalModel(embedder)
retrieval_model.add_snippets(snippets)
config = {"weights": weights}
queries = load_jsonl(Path('nbs/datasets/prompts.jsonl'))
retrieval_model.log_query_results(queries)

# In[ ]:




