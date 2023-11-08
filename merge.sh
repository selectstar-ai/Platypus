python llm-processing/train/Platypus/merge.py \
    --base_model_name_or_path hyunseoki/ko-en-llama2-13b \
    --peft_model_path ckpt/llama2_tmt-13b-v2/checkpoint-100 \
    --output_dir llama2_tmt-13b-v2 \
    --push True