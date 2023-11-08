from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()

def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    
    model.push_to_hub(f"{args.output_dir}")
    tokenizer.push_to_hub(f"{args.output_dir}")
    print(f"Model pushed to {args.output_dir}")

if __name__ == "__main__" :
    main()