# LLM Classification Code

This directory consists of the codes to optimally fine-tune LLM model using Accelerate with DeepSpeed.

## Setup
```
pip install -r requirements.txt
```

## Usage
### Fine-Tuning

To optimally use codebase please run it using accelerate with deepspeed.
Bash command for the same is:
```
accelerate launch --config_file deepspeed_config.yaml train.py
```

### Model conversion to bin format

##### Steps
1. Change the directory to the checkpoint directory (i.e. `cd /path/to/checkpoint_dir`)
2. Run `./zero_to_fp32.py . pytorch_model.bin`

After successfully running the above command, we are good to use it for inference.
