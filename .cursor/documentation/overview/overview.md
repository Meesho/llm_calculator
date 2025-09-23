# Overview

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)

</details>



This document provides an overview of the LLM Calculator repository, a dual-purpose system designed for optimizing and executing fine-tuning workflows for Llama2-7b models. The repository contains two main packages: a runtime calculator for parameter optimization and a complete training pipeline with performance optimizations.

For detailed information about the runtime calculation functionality, see [LLM Runtime Calculator Package](#2). For information about the actual training implementation, see [LLM Training Codebase Package](#3).

## System Purpose

The LLM Calculator repository addresses two critical needs in LLM fine-tuning:

1. **Runtime Optimization**: Calculates optimal training parameters to minimize computational time and resources
2. **Efficient Training**: Provides a production-ready fine-tuning pipeline with advanced optimization techniques

The system is specifically designed for fine-tuning the `NousResearch/Llama-2-7b-hf` model using parameter-efficient techniques and memory optimization strategies.

## Architecture Overview

The following diagram shows the high-level architecture of the LLM Calculator system:

### System Architecture
```mermaid
graph TB
    subgraph "optimal_llm_calculator"
        CALC_MAIN["optimal_runtime_calculator.py"]
        CALC_DATA["sample-dataset.csv"]
        CALC_REQ["requirements.txt"]
    end
    
    subgraph "optimal_llm_codebase"
        TRAIN_MAIN["train.py"]
        TRAIN_UTILS["utils.py"]
        TRAIN_CONFIG["deepspeed_config.yaml"]
        TRAIN_PATCH["llama_flash_attn_monkey_patch.py"]
        TRAIN_DATA["bitcoin-sentiment-dataset.csv"]
        TRAIN_REQ["requirements.txt"]
    end
    
    subgraph "External Systems"
        HF_MODELS["NousResearch/Llama-2-7b-hf"]
        DEEPSPEED["DeepSpeed ZeRO-3"]
        FLASH_ATTN["Flash Attention"]
    end
    
    CALC_MAIN --> TRAIN_CONFIG
    CALC_DATA --> CALC_MAIN
    
    TRAIN_CONFIG --> TRAIN_MAIN
    TRAIN_UTILS --> TRAIN_MAIN
    TRAIN_PATCH --> TRAIN_MAIN
    TRAIN_DATA --> TRAIN_UTILS
    
    HF_MODELS --> TRAIN_UTILS
    DEEPSPEED --> TRAIN_CONFIG
    FLASH_ATTN --> TRAIN_PATCH
    
    CALC_MAIN -.->|"Optimal Parameters"| TRAIN_CONFIG
```

*Sources: README.md, system architecture analysis*

## Two-Phase Workflow

The system implements a two-phase approach to LLM fine-tuning optimization:

### Complete Workflow Process
```mermaid
flowchart TD
    START["User Input: Model & Dataset Requirements"] --> PHASE1["Phase 1: Parameter Optimization"]
    
    PHASE1 --> CALC_SCRIPT["optimal_runtime_calculator.py"]
    CALC_SCRIPT --> SAMPLE_DATA["sample-dataset.csv"]
    CALC_SCRIPT --> TOKENIZER["LlamaTokenizerFast"]
    
    TOKENIZER --> CALC_ENGINE["datapoints_in_1_step calculation"]
    CALC_ENGINE --> TIME_EST["Time estimation formulas"]
    TIME_EST --> OPTIMAL_PARAMS["Optimal batch_size, epochs, etc."]
    
    OPTIMAL_PARAMS --> PHASE2["Phase 2: Training Execution"]
    
    PHASE2 --> DS_CONFIG["deepspeed_config.yaml configuration"]
    DS_CONFIG --> TRAIN_LAUNCH["accelerate launch train.py"]
    
    TRAIN_LAUNCH --> FLASH_PATCH["llama_flash_attn_monkey_patch.py"]
    FLASH_PATCH --> MODEL_PREP["utils.py::create_and_prepare_model"]
    TRAIN_LAUNCH --> DATA_PREP["utils.py::create_datasets"]
    
    MODEL_PREP --> LORA_MODEL["LlamaForSequenceClassification + LoRA"]
    DATA_PREP --> TOKENIZED_DATA["Tokenized sentiment dataset"]
    
    LORA_MODEL --> CUSTOM_TRAINER["CustomTrainer with weighted loss"]
    TOKENIZED_DATA --> CUSTOM_TRAINER
    
    CUSTOM_TRAINER --> CHECKPOINTS["Model checkpoints"]
    CHECKPOINTS --> PYTORCH_MODEL["pytorch_model.bin"]
    
    subgraph "Calculator Components"
        CALC_SCRIPT
        SAMPLE_DATA
        TOKENIZER
        CALC_ENGINE
        TIME_EST
        OPTIMAL_PARAMS
    end
    
    subgraph "Training Components"  
        DS_CONFIG
        TRAIN_LAUNCH
        FLASH_PATCH
        MODEL_PREP
        DATA_PREP
        LORA_MODEL
        TOKENIZED_DATA
        CUSTOM_TRAINER
    end
```

*Sources: System workflow analysis, optimal_runtime_calculator.py, train.py, utils.py*

## Key Components

The repository is organized into several key components that work together to provide the complete optimization and training pipeline:

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Runtime Calculator** | Parameter optimization and time estimation | `optimal_runtime_calculator.py`, `sample-dataset.csv` |
| **Training Pipeline** | Core fine-tuning logic with custom trainer | `train.py`, `utils.py` |
| **Performance Optimization** | Memory and speed enhancements | `deepspeed_config.yaml`, `llama_flash_attn_monkey_patch.py` |
| **Training Data** | Sentiment classification dataset | `bitcoin-sentiment-dataset.csv` |

### Component Relationships
```mermaid
graph LR
    subgraph "Data Flow"
        SAMPLE["sample-dataset.csv"] --> CALC["optimal_runtime_calculator.py"]
        BITCOIN["bitcoin-sentiment-dataset.csv"] --> UTILS["utils.py"]
    end
    
    subgraph "Configuration Flow"
        CALC --> CONFIG["deepspeed_config.yaml"]
        CONFIG --> TRAIN["train.py"]
    end
    
    subgraph "Optimization Chain"
        PATCH["llama_flash_attn_monkey_patch.py"] --> UTILS
        UTILS --> TRAIN
        CONFIG --> TRAIN
    end
    
    subgraph "Model Pipeline"
        UTILS --> MODEL["create_and_prepare_model"]
        UTILS --> DATASET["create_datasets"] 
        MODEL --> TRAINER["CustomTrainer"]
        DATASET --> TRAINER
        TRAIN --> TRAINER
    end
```

*Sources: File structure analysis, component dependency mapping*

## Performance Optimizations

The system incorporates multiple optimization techniques to handle large model fine-tuning efficiently:

- **DeepSpeed ZeRO Stage 3**: Memory partitioning across GPUs for large model support
- **Flash Attention**: Optimized attention mechanism for compatible hardware
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning approach
- **8-bit Quantization**: Memory reduction through model quantization
- **Gradient Checkpointing**: Trade compute for memory during backpropagation

For specific implementation details of the runtime calculator, see [Runtime Calculator Script](#2.1). For training pipeline specifics, see [Training Pipeline](#3.1). For performance optimization configurations, see [DeepSpeed Configuration](#3.2) and [Flash Attention Optimization](#3.3).

*Sources: README.md, system architecture analysis*
