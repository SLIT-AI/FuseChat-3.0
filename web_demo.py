import streamlit as st
import torch
import time
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/llm/FuseAI")
parser.add_argument("--model-name", type=str, default="FuseChat-Llama-3.1-8B-Instruct")
parser.add_argument("--title-name", type=str, default="FuseChat-Llama-3.1-8B-Instruct")
args = parser.parse_args()

# App title
st.set_page_config(page_title=f"😶‍🌫️FuseAI {args.title_name}")


@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        f"{args.model_path}/{model_name}",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        f"{args.model_path}/{model_name}",
        device_map="cuda",
        # load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


with st.sidebar:
    st.title(f'😶‍🌫️FuseAI {args.title_name}')
    st.write(f'This chatbot is created using the {args.title_name} model.')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.6, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    top_k = st.sidebar.slider('top_k', min_value=1, max_value=1000, value=1000, step=1)
    repetition_penalty = st.sidebar.slider('repetition penalty', min_value=1.0, max_value=2.0, value=1.2, step=0.05)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=4096, value=2048, step=8)

with st.spinner('loading model..'):
    model, tokenizer = load_model(args.model_name)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def set_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
# Create a list of candidate questions
candidate_questions = ["Can you tell me a joke?", "Write a quicksort code in Python.", "Write a poem about love in Shakespearean tone."]
# Display the chat interface with a list of clickable question buttons
for question in candidate_questions:
    st.sidebar.button(label=question, on_click=set_query, args=[question])

def clear_chat_history():
    st.session_state.messages = []
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

@torch.no_grad()
def generate_fusechat_response():
    conversations=[]
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            conversations.append({"role":"user", "content":dict_message["content"]})
        else:
            conversations.append({"role":"assistant", "content":dict_message["content"]})
    string_dialogue = tokenizer.apply_chat_template(conversations,tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer(string_dialogue,
                          return_tensors="pt").input_ids.to('cuda')
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_length,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
    return "".join(outputs)

# User-provided prompt
if prompt := st.chat_input("Hello there! How are you doing?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_fusechat_response()
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                time.sleep(0.001)
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)