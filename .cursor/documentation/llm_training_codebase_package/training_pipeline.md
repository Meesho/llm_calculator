# Training Pipeline

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [optimal_llm_codebase/train.py](optimal_llm_codebase/train.py)
- [optimal_llm_codebase/utils.py](optimal_llm_codebase/utils.py)

</details>



## Purpose and Scope

The Training Pipeline implements the complete fine-tuning workflow for Llama2-7b models using parameter-efficient techniques. This document covers the main training script, utility functions, and the orchestration of model preparation, dataset creation, and training execution within the `optimal_llm_codebase` package.

For information about the runtime calculation that determines optimal parameters for this pipeline, see [Runtime Calculator Script](#2.1). For details about the DeepSpeed configuration used by this pipeline, see [DeepSpeed Configuration](#3.2).

## Pipeline Architecture Overview

The training pipeline consists of two primary components that work together to execute fine-tuning with memory and performance optimizations:

```mermaid
graph TB
    subgraph "Training Pipeline Components"
        TRAIN["train.py<br/>Main Training Script"]
        UTILS["utils.py<br/>Utility Functions"]
    end
    
    subgraph "Core Functions"
        MAIN["main()<br/>Pipeline Orchestrator"]
        CUSTOM["CustomTrainer<br/>Weighted Loss Training"]
        METRICS["compute_metrics()<br/>Evaluation Logic"]
    end
    
    subgraph "Model Preparation"
        CREATE_MODEL["create_and_prepare_model()<br/>LoRA + Flash Attention Setup"]
        FLASH_PATCH["replace_llama_attn_with_flash_attn()<br/>Performance Optimization"]
        PEFT_CONFIG["LoraConfig<br/>Parameter-Efficient Setup"]
    end
    
    subgraph "Data Processing"
        CREATE_DATASETS["create_datasets()<br/>CSV Loading + Tokenization"]
        TOKENIZE["tokenize_function()<br/>Text to Token Conversion"]
        DATASET_SPLIT["Train/Eval Split<br/>Bitcoin Sentiment Data"]
    end
    
    subgraph "Training Infrastructure"
        CALLBACK["SavePeftDeepSpeedModelCallback<br/>Checkpoint Management"]
        TRAINER_ARGS["TrainingArguments<br/>Hyperparameter Configuration"]
        DEEPSPEED["DeepSpeed Integration<br/>Memory Optimization"]
    end
    
    TRAIN --> MAIN
    TRAIN --> CUSTOM
    TRAIN --> METRICS
    
    UTILS --> CREATE_MODEL
    UTILS --> CREATE_DATASETS
    UTILS --> CALLBACK
    
    CREATE_MODEL --> FLASH_PATCH
    CREATE_MODEL --> PEFT_CONFIG
    
    CREATE_DATASETS --> TOKENIZE
    CREATE_DATASETS --> DATASET_SPLIT
    
    MAIN --> CREATE_MODEL
    MAIN --> CREATE_DATASETS
    MAIN --> CUSTOM
    MAIN --> CALLBACK
```

**Sources:** [optimal_llm_codebase/train.py:1-314](), [optimal_llm_codebase/utils.py:1-149]()

## Main Training Script Components

The `train.py` script serves as the entry point and orchestrator for the entire training process. It implements several key components:

### Command Line Interface

The script accepts comprehensive configuration through command line arguments covering model parameters, training hyperparameters, and infrastructure settings:

| Parameter Category | Key Arguments | Purpose |
|-------------------|---------------|---------|
| Model Configuration | `model_name`, `lora_r`, `lora_alpha`, `lora_dropout` | Define base model and LoRA settings |
| Training Batch Settings | `per_device_train_batch_size`, `gradient_accumulation_steps` | Control memory usage and effective batch size |
| Optimization | `optim`, `max_grad_norm`, `warmup_ratio`, `lr_scheduler_type` | Configure training dynamics |
| Infrastructure | `bf16`, `use_gradient_checkpointing`, `output_dir` | Enable performance optimizations |

**Sources:** [optimal_llm_codebase/train.py:182-309]()

### CustomTrainer Implementation

The pipeline implements a specialized trainer class that extends the base `Trainer` with custom loss computation:

```mermaid
graph LR
    subgraph "CustomTrainer Class"
        COMPUTE_LOSS["compute_loss()<br/>Weighted CrossEntropyLoss"]
        AUTOCAST["torch.autocast('cuda')<br/>Mixed Precision"]
        LOSS_WEIGHTS["Class Weights: [2.35, 0.64]<br/>Handle Imbalanced Data"]
    end
    
    subgraph "Training Process"
        FORWARD["Forward Pass<br/>Model Inference"]
        LOGITS["Extract Logits<br/>Classification Outputs"]
        WEIGHTED_LOSS["Apply Weighted Loss<br/>Address Class Imbalance"]
    end
    
    COMPUTE_LOSS --> AUTOCAST
    AUTOCAST --> FORWARD
    FORWARD --> LOGITS
    LOGITS --> LOSS_WEIGHTS
    LOSS_WEIGHTS --> WEIGHTED_LOSS
```

The custom loss function addresses class imbalance in the sentiment dataset by applying weights of `[2.35, 0.64]` to the CrossEntropyLoss, emphasizing the minority class.

**Sources:** [optimal_llm_codebase/train.py:62-81]()

### Evaluation Metrics

The training pipeline implements precision-based evaluation through the `compute_metrics` function, which integrates with the HuggingFace `evaluate` library:

**Sources:** [optimal_llm_codebase/train.py:26-54]()

## Utility Functions and Model Preparation

The `utils.py` module provides critical infrastructure for model and dataset preparation:

### Model Creation and Optimization Pipeline

```mermaid
flowchart TD
    START["create_and_prepare_model()"] --> FLASH["replace_llama_attn_with_flash_attn()<br/>Apply Flash Attention Patch"]
    
    FLASH --> LOAD["LlamaForSequenceClassification.from_pretrained()<br/>Load Base Model"]
    
    LOAD --> LORA_CONFIG["LoraConfig<br/>r=16, alpha=64, dropout=0.1<br/>target_modules=['q_proj', 'v_proj']"]
    
    LORA_CONFIG --> GRAD_CHECK{"use_gradient_checkpointing?"}
    GRAD_CHECK -->|Yes| ENABLE_GRAD["model.gradient_checkpointing_enable()"]
    GRAD_CHECK -->|No| PREPARE
    ENABLE_GRAD --> PREPARE
    
    PREPARE["prepare_model_for_kbit_training()<br/>Prepare for Parameter-Efficient Training"]
    
    PREPARE --> PEFT["get_peft_model(model, peft_config)<br/>Apply LoRA Adapters"]
    
    PEFT --> CASTING["peft_module_casting_to_bf16()<br/>Optimize Data Types"]
    
    CASTING --> TOKENIZER["LlamaTokenizerFast.from_pretrained()<br/>Setup Tokenization"]
    
    TOKENIZER --> RETURN["Return (model, tokenizer)"]
```

The model preparation process integrates multiple optimization techniques:
- **Flash Attention**: Applied via monkey patching for memory efficiency
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning with rank 16
- **Gradient Checkpointing**: Memory optimization at the cost of computation
- **Mixed Precision**: BFloat16 casting for compatible modules

**Sources:** [optimal_llm_codebase/utils.py:91-135](), [optimal_llm_codebase/utils.py:138-149]()

### Dataset Creation and Processing

The dataset pipeline transforms raw CSV data into tokenized training and evaluation sets:

```mermaid
graph TD
    CSV["bitcoin-sentiment-dataset.csv<br/>Raw Sentiment Data"] --> REPEAT["np.repeat(df.values, 3, axis=0)<br/>Triple Dataset Size"]
    
    REPEAT --> RENAME["Rename Columns<br/>output → labels, input → text"]
    
    RENAME --> CONVERT["Text to Numeric Labels<br/>Positive: 1, Negative: 0"]
    
    CONVERT --> SPLIT["Train/Test Split<br/>Train: 0-10000, Test: 10000+"]
    
    SPLIT --> TOKENIZE["tokenize_function()<br/>truncation=True, padding=True<br/>max_length=550"]
    
    TOKENIZE --> DATASETS["Return tokenized datasets<br/>with input_ids, attention_mask"]
```

**Sources:** [optimal_llm_codebase/utils.py:59-88](), [optimal_llm_codebase/utils.py:52-56]()

## Training Execution Flow

The main training loop orchestrates the complete pipeline through several phases:

### Initialization and Setup

```mermaid
sequenceDiagram
    participant MAIN as "main()"
    participant MODEL as "create_and_prepare_model()"
    participant DATA as "create_datasets()"
    participant TRAINER as "CustomTrainer"
    participant CALLBACK as "SavePeftDeepSpeedModelCallback"
    
    MAIN->>MODEL: Initialize model with LoRA + Flash Attention
    MODEL-->>MAIN: Return (model, tokenizer)
    
    MAIN->>DATA: Create tokenized datasets
    DATA-->>MAIN: Return (train_dataset, eval_dataset)
    
    MAIN->>MAIN: Calculate num_of_steps_in_1_epoch
    Note over MAIN: steps = dataset_size / (batch_size * grad_acc * 8)
    
    MAIN->>TRAINER: Initialize with TrainingArguments
    MAIN->>CALLBACK: Add DeepSpeed PEFT callback
    
    MAIN->>TRAINER: trainer.train()
    TRAINER-->>MAIN: training_result
    
    MAIN->>MAIN: Save final model with DeepSpeed state
```

**Sources:** [optimal_llm_codebase/train.py:83-180]()

### Training Configuration

The pipeline calculates training steps dynamically based on dataset size and distributed training parameters:

```
num_of_steps_in_1_epoch = train_dataset.shape[0] / (
    per_device_train_batch_size * gradient_accumulation_steps * 8
)
```

Key training arguments include:
- **Mixed Precision**: BFloat16 enabled for performance
- **Gradient Checkpointing**: Memory optimization
- **Evaluation Strategy**: Step-based evaluation every 3 steps
- **Optimizer**: Adafactor for memory-efficient optimization

**Sources:** [optimal_llm_codebase/train.py:90-123]()

## Model Checkpointing and Persistence

The training pipeline implements sophisticated checkpointing through the `SavePeftDeepSpeedModelCallback` class:

### DeepSpeed-Compatible Checkpointing

```mermaid
graph LR
    subgraph "Callback Execution"
        STEP_CHECK["on_step_end()<br/>Check step % save_steps == 0"]
        WAIT_ALL["trainer.accelerator.wait_for_everyone()<br/>Synchronize Processes"]
        GET_STATE["accelerator.get_state_dict(trainer.deepspeed)<br/>Collect Distributed State"]
        UNWRAP["accelerator.unwrap_model(trainer.deepspeed)<br/>Extract Model"]
        SAVE["unwrapped_model.save_pretrained()<br/>Persist Checkpoint"]
    end
    
    STEP_CHECK --> WAIT_ALL
    WAIT_ALL --> GET_STATE
    GET_STATE --> UNWRAP
    UNWRAP --> SAVE
```

The callback ensures proper state collection across distributed training processes and saves only on the main process to avoid conflicts.

**Sources:** [optimal_llm_codebase/utils.py:22-49]()

### Final Model Persistence

After training completion, the pipeline performs a final model save with complete state synchronization and exports training metrics to CSV format for analysis.

**Sources:** [optimal_llm_codebase/train.py:166-180]()
