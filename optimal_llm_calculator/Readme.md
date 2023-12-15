# Optimal RunTime Calculator

## Introduction
This calculator calculates the time taken by the data to completely fine-tune the `Llama2-7b` model in an optimal manner.
The time is reduced to greater extent when we use flash attention which needs A100 machine. 
Hence it is better to use `A100` machine. 

## Setup
```
pip install -r requirements.txt
```

## How to Run:
```
python optimal_runtime_calculator.py \
         --hf_model_name_or_path NousResearch/Llama-2-7b-hf \
         --data_path ./sample-dataset.csv \
         --data_field 'text' \
         --number_of_datapoints_to_keep_in_eval 5 \
         --batch_size 8 \
         --gradient_accumulation_step 64 \
         --num_of_epochs 1 \
         --save_steps 3 \
         --eval_steps 3 \
         --max_length 1100 \
         --padding True \
         --truncation True
```