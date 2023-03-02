#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import numpy as np
import sys

from codesearch.ensemble.ensemble_embedder import EnsembleEmbedder
from codesearch.encoders import BasicEncoder
from codesearch import embedding_pretraining
from codesearch.embedding_pretraining import train_fasttext_model_from_snippets, load_fasttext_model
from codesearch.utils import SaveableFunction
from codesearch.data import load_snippet_collection, EVAL_DATASETS, SNIPPET_COLLECTIONS, eval_datasets_from_regex
from codesearch.ncs.ncs_embedder import TfidfCodeEmbedder, NcsEmbedder
from codesearch.evaluation import evaluate_and_dump
from codesearch.embedding_retrieval import EmbeddingRetrievalModel

def create_retrieval_model(model_name):
    start = time.time()
    print(f"start={start}")
    sys.path.append('../')
    import processing


    fast_text_checkpoint = 'codesearch/nbs/ncs/ncs_pls_work'# os.environ.get("fast_text_checkpoint", None)
    #fast_text_checkpoint = None

    snippets_collection = os.environ.get("snippets_collection", "so-ds-feb20")
    train_snippets_collection = os.environ.get("train_snippets_collection", "so-ds-feb20")
    valid_dataset = os.environ.get("valid_dataset", None)
    test_dataset = os.environ.get("test_dataset", "so-ds-feb20-test")

    text_overrides = json.loads(os.environ.get("text_overrides", "{}"))
    code_overrides = json.loads(os.environ.get("code_overrides", "{}"))
    fast_text_overrides = json.loads(os.environ.get("fast_text_overrides", "{}"))
    zip_fn_name = os.environ.get("zip_fn", "zip_descr_end")
    output_dir = os.environ.get("output_dir", "")

    print(f"fast_text_checkpoint={fast_text_checkpoint}")
    print(f"snippets_collection={snippets_collection}")
    print(f"text_overrides={text_overrides}\n code_overrides={code_overrides}\n fast_text_overrides={fast_text_overrides} zip_fn_name={zip_fn_name}")

    if valid_dataset and valid_dataset not in EVAL_DATASETS and valid_dataset not in SNIPPET_COLLECTIONS:
        raise ValueError()
    test_datasets = eval_datasets_from_regex(test_dataset)
    snippets = load_snippet_collection(snippets_collection)
    train_snippets = load_snippet_collection(train_snippets_collection)
    if model_name == 'ncs':
        if fast_text_checkpoint:
            print("here")
            model, enc = load_fasttext_model(fast_text_checkpoint)
            print("Loaded fast text checkpoint")

        else:
            enc = BasicEncoder(text_preprocessing_params=text_overrides, code_preprocessing_params=code_overrides)
            zip_fn = getattr(sys.modules[embedding_pretraining.__name__], zip_fn_name)
            model = train_fasttext_model_from_snippets(train_snippets, enc, zip_fn, fast_text_overrides,
                                                       "", save=True)

        tfidf_model = TfidfCodeEmbedder.create_tfidf_model(enc, model, snippets)
        embedder = NcsEmbedder(model, enc, tfidf_model)
        retrieval_model = EmbeddingRetrievalModel(embedder)
        retrieval_model.add_snippets(snippets)
    elif model_name == 'ensemble':
        model_names = [ncs_model,
                       "../tuse/pacsv1/use5-act=relu_sigmoid-sim=cosine-negsamples=5-lr=0.0001-dropout=0.0-date=87/use_steps=1300", ]

        for weights in [[.4, 1.], [.5, 1.], [.6, 1.], [.7, 1.]]:
            embedder = EnsembleEmbedder(model_names, weights)
            retrieval_model = EmbeddingRetrievalModel(embedder)
            retrieval_model.add_snippets(snippets)

    print("got the retrieval model")
    print("printing my current working directory")
    print(os.getcwd())
    #sample_queries = processing.ds_processing()
    #print("got sample queries")
    #sample_queries = ["train a tensorflow model", "plot a bar chart", "merge two dataframes", "sort a list", "read a pandas dataframe from a file", "plot an image"]
    #print(sample_queries)
    #results = retrieval_model.log_query_results(sample_queries)
    return retrieval_model
