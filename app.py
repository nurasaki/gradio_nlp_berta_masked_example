import gradio as gr
import os

 

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
from torch.nn.functional import softmax
import pandas as pd



# save your HF API token from https:/hf.co/settings/tokens as an env variable to avoid rate limiting
auth_token = os.getenv("auth_token")





print("========================================================================")
print("Starting ... gradio_demo_nlp_autocomplete/app.py")
print("AUTH TOKEN:", auth_token)


# load a model from https://hf.co/models as an interface, then use it as an api 
# you can remove the api_key parameter if you don't care about rate limiting. 
# api = gr.Interface.load(, api_key=auth_token,)


model_ref = "projecte-aina/roberta-base-ca-v2"
tokenizer = AutoTokenizer.from_pretrained(model_ref)
model = AutoModelForMaskedLM.from_pretrained(model_ref)

def get_topk(text, tokenizer, model, k):

    print("Get top K,", text)

    # Tokenize
    # ==========================================================================================
    tokenizer_kwargs = dict(padding='longest', return_token_type_ids=False, return_tensors="pt")
    inputs = tokenizer(text, **tokenizer_kwargs).to("cpu")
    input_ids = inputs.input_ids

    
    # Get model outputs and probabilities
    # ==========================================================================================
    # logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = model.to("cpu")(**inputs).logits
    probs = softmax(logits, dim=2)
    
    
    # Index ok <mask> (ojo només funciona quan hi ha 1 MASK)
    # ==========================================================================================
    row_idx, mask_idx = torch.where(input_ids.to("cpu") == tokenizer.mask_token_id)

    return probs[row_idx, mask_idx].topk(k), mask_idx


def generate_output(text, k):

    # lines = print_topk(text, tokenizer, model, k=10)

    (values, indices), input_idx = get_topk(text, tokenizer, model, int(k))

    for mask_vals, mask_indices, input_idx in zip(values, indices, input_idx):
        labels = {tokenizer.decode(ind): val.item()
                  for val, ind in zip(mask_vals, mask_indices)}

    return labels


md_text ="""
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



"""

examples = [
    "La meva mare es diu <mask>.",
    "La meva mare treballa de <mask>.",
    "El meu fill es diu <mask>.",
    "El teu pare treballa de <mask>.",
]



with gr.Blocks() as demo:
    gr.Markdown(md_text)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox("La meva mare es diu <mask>.", label="Masked text")
            k = gr.Number(value=10, label="Num. results")
            btn = gr.Button("Generate")
            
        with gr.Column():
            out_label = gr.Label(label="Results")
    
    
    btn.click(generate_output, inputs=[text, k], outputs=[out_label])
    gr.Examples(examples, inputs=[text])

# if __name__ == "__main__":
demo.launch(favicon_path="favicon.png")
