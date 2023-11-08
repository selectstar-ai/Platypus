import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import gc

import streamlit as st

# Check for available device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def inference(
    instruction: str = "안녕, 넌 누구니?",
    input_text: str = "",
    base_model: str = "jiwoochris/ko-llama2-v1",
    lora_weights: str = "",
    prompt_template: str = "alpaca",
    output_csv_path: str = ""
):

    interactive_qa_session(base_model, lora_weights, prompt_template, output_csv_path)

def load_model(base_model, lora_weights, device, load_8bit=False):
    if device in ["cuda", "mps"]:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    return model

def evaluate_single(prompter, prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generation_output = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1,
                                       max_new_tokens=2048, temperature=0.15, top_p=0.95)
    
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    resp = prompter.get_response(output)
    print(resp)
    return resp

def interactive_qa_session(base_model, lora_weights, prompt_template, output_csv_path):
    # Load model and tokenizer outside the loop for efficiency
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = load_model(base_model, lora_weights, device, load_8bit=False)
    
    while True:
        # Get user input
        instruction = input("Ask a question (or type 'exit' to quit): ")
        if instruction.lower() == 'exit':
            break

        # Generate the prompt
        prompt = prompter.generate_prompt(instruction)
        
        # Evaluate the prompt
        result = evaluate_single(prompter, prompt, model, tokenizer)
        
        # Output the result
        print(f"Answer: {result}")
        
        # Optionally, save the Q&A pair to CSV
        if output_csv_path:
            append_to_csv(output_csv_path, instruction, result)
            
def append_to_csv(output_csv_path, question, answer):
    # This function will append the Q&A pair to a CSV file
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([question, answer])


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(interactive_qa_session)
