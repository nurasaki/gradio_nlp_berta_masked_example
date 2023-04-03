import gradio as gr
import os

# save your HF API token from https:/hf.co/settings/tokens as an env variable to avoid rate limiting
auth_token = os.getenv("auth_token")




print("========================================================================")
print("Starting ... gradio_demo_nlp_autocomplete/app.py")
print("AUTH TOKEN:", auth_token)


# load a model from https://hf.co/models as an interface, then use it as an api 
# you can remove the api_key parameter if you don't care about rate limiting. 
api = gr.Interface.load("huggingface/projecte-aina/roberta-base-ca-v2", api_key=auth_token,)






def complete_with_gpt(text):

    print("------------------------------------------------------------------------")
    print("type(api):", type(api) )
    print("Api:", api, "\n" )



    print("------------------------------------------------------------------------")
    print("text:")
    print(text)
    print("------------------------------------------------------------------------")
    print("text[:-50]:")
    print(text[:-50])
    print("------------------------------------------------------------------------")
    print("api(text):")
    print(api(text))
    print("------------------------------------------------------------------------")
    print("text[-50:]:")
    print(text[-50:])
    print("------------------------------------------------------------------------")
    print("api(text[-50:]")
    print(api(text[-50:]))
    print("------------------------------------------------------------------------")
    

    return text[:-50] + api(text[-50:])


with gr.Blocks() as demo:
    
    print("------------------------------------------------------------------------")
    print("with gr.Blocks")

    textbox = gr.Textbox(placeholder="Type here...", lines=4)
    btn = gr.Button("Autocomplete")
    
    print("textbox:", textbox)

    # define what will run when the button is clicked, here the textbox is used as both an input and an output
    btn.click(fn=complete_with_gpt, inputs=textbox, outputs=textbox, queue=False)

demo.launch(favicon_path="favicon.png")


# /Users/nurasaki/miniforge3/envs/conda_tfg_clone/lib/python3.8/site-packages/gradio/interface.py:93: 
# UserWarning: gr.Intrerface.load() will be deprecated. Use gr.load() instead.
# warnings.warn("gr.Intrerface.load() will be deprecated. Use gr.load() instead.")