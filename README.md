---
title: Gradio NLP - Berta Masked Example
emoji: 💻
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 3.23.0
app_file: app.py
pinned: false
---


# Masked Language Modeling Example

by [nurasaki](https://huggingface.co/spaces/nurasaki)  

* Space : [https://huggingface.co/spaces/nurasaki/gradio_nlp_berta_masked_example](https://huggingface.co/spaces/nurasaki/gradio_nlp_berta_masked_example)
* Model used: Catalan BERTa-v2 (roberta-base-ca-v2) base model
* Hugginface link: [https://huggingface.co/projecte-aina/roberta-base-ca-v2](https://huggingface.co/projecte-aina/roberta-base-ca-v2)

<br>

## Model description

The **roberta-base-ca-v2** is a transformer-based masked language model for the Catalan language. 

It is based on the [RoBERTA](https://github.com/pytorch/fairseq/tree/master/examples/roberta) base model and has been trained on a medium-size corpus collected from publicly available corpora and crawlers.

<br>

## Usage

The model accepts an input text with a *mask* (for example, "La meva mare es diu \<mask\>.") and generates the *k* most probable words that could fill the *mask* position in the sentence.  

Choose one of the provided examples or enter your own masked text.

<br>

