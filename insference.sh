python llm-processing/train/Platypus/inference.py \
    --base_model "hyunseoki/ko-en-llama2-13b" \
    --load_8bit False \
    --lora_weights "llama2-13b-1/checkpoint-180" \
    --prompt_template "alpaca" \
    --csv_path "data/eval/quality_eval_data.csv" \
    --output_csv_path "data/eval/result-180.csv"