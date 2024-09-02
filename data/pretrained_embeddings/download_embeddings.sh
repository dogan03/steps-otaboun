#!/bin/bash

# Script for downloading pretrained word embeddings.
# Note that we are NOT DISTRIBUTING these models with our code due to licensing reasons.
# Skip downloading a model by commenting out the corresponding commands.

# BERT Base Turkish Cased
mkdir bert-base-turkish-cased
curl https://huggingface.co/dbmdz/bert-base-turkish-cased/resolve/main/config.json -o bert-base-turkish-cased/config.json --create-dirs
curl https://huggingface.co/dbmdz/bert-base-turkish-cased/resolve/main/vocab.txt -o bert-base-turkish-cased/vocab.txt
curl https://cdn-lfs.huggingface.co/dbmdz/bert-base-turkish-cased/0a7b7e19eeb5758dd264baf230e72eecb46cf6b7d278cb1e87a11a0c92480f12 -o bert-base-turkish-cased/pytorch_model.bin
