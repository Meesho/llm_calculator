# Training Dependencies

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [optimal_llm_codebase/requirements.txt](optimal_llm_codebase/requirements.txt)

</details>



This document covers the external library dependencies required for the LLM fine-tuning training pipeline in the `optimal_llm_codebase` package. It details the specific versions, installation requirements, and compatibility considerations for the distributed training setup.

For information about the calculator's dependencies, see the Runtime Calculator documentation. For the actual training implementation that uses these dependencies, see [Training Pipeline](#3.1).

## Dependency Categories

The training pipeline requires dependencies in four main categories: distributed training, model optimization, monitoring, and mathematical operations.

### Core Training Framework Dependencies

The foundation of the training system relies on Hugging Face ecosystem libraries with specific version constraints:

| Library | Version/Source | Purpose |
|---------|----------------|---------|
| `transformers` | 4.31.0 (pinned) | Base model loading and tokenization |
| `accelerate` | Git (latest) | Distributed training orchestration |
| `peft` | Git (latest) | Parameter-efficient fine-tuning (LoRA) |
| `trl` | PyPI (latest) | Transformer reinforcement learning utilities |

### Performance Optimization Dependencies

Memory and compute optimization libraries are installed from source repositories:

| Library | Source | Integration Point |
|---------|--------|-------------------|
| `flash-attention` | Dao-AILab/flash-attention | `llama_flash_attn_monkey_patch.py` |
| `DeepSpeed` | microsoft/DeepSpeed | `deepspeed_config.yaml` |
| `einops` | PyPI | Tensor operation utilities |

### Monitoring and Evaluation Dependencies

| Library | Purpose |
|---------|---------|
| `wandb` | Training metrics logging and visualization |
| `evaluate` | Model performance evaluation metrics |

Sources: [optimal_llm_codebase/requirements.txt:1-16]()

## Version Compatibility Architecture

```mermaid
graph TB
    subgraph "Version Constraint Issues"
        DS_ISSUE["DeepSpeed Issues #4229, #4194"]
        TRANSFORMERS_PIN["transformers==4.31.0"]
        GIT_DEPS["Git-based Dependencies"]
    end
    
    subgraph "Core Dependencies"
        TRANSFORMERS["transformers==4.31.0"]
        ACCELERATE["git+https://github.com/huggingface/accelerate"]
        PEFT["git+https://github.com/huggingface/peft"]
        FLASH["git+https://github.com/Dao-AILab/flash-attention"]
        DEEPSPEED["git+https://github.com/microsoft/DeepSpeed"]
    end
    
    subgraph "Auxiliary Dependencies"
        TRL["trl"]
        WANDB["wandb"]
        EVALUATE["evaluate"]
        EINOPS["einops"]
    end
    
    DS_ISSUE --> TRANSFORMERS_PIN
    DS_ISSUE --> DEEPSPEED
    
    TRANSFORMERS --> ACCELERATE
    TRANSFORMERS --> PEFT
    ACCELERATE --> DEEPSPEED
    PEFT --> FLASH
    
    style DS_ISSUE fill:#ffebee
    style TRANSFORMERS_PIN fill:#fff3e0
```

The dependency graph shows a critical version pinning decision where `transformers` is locked to version 4.31.0 due to compatibility issues with DeepSpeed, while other core libraries are installed from git to get the latest features.

Sources: [optimal_llm_codebase/requirements.txt:1-6]()

## Dependency Integration with Training Components

```mermaid
graph TB
    subgraph "requirements.txt Dependencies"
        TRANSFORMERS_DEP["transformers==4.31.0"]
        ACCELERATE_DEP["accelerate (git)"]
        PEFT_DEP["peft (git)"]
        FLASH_DEP["flash-attention (git)"]
        DEEPSPEED_DEP["DeepSpeed (git)"]
        TRL_DEP["trl"]
        WANDB_DEP["wandb"]
        EVALUATE_DEP["evaluate"]
        EINOPS_DEP["einops"]
    end
    
    subgraph "Training Code Integration"
        TRAIN_PY["train.py"]
        UTILS_PY["utils.py"]
        DEEPSPEED_CONFIG["deepspeed_config.yaml"]
        FLASH_PATCH["llama_flash_attn_monkey_patch.py"]
    end
    
    subgraph "Specific Functions"
        CREATE_MODEL["create_and_prepare_model()"]
        CREATE_DATASETS["create_datasets()"]
        CUSTOM_TRAINER["CustomTrainer"]
        REPLACE_ATTN["replace_llama_attn_with_flash_attn()"]
    end
    
    TRANSFORMERS_DEP --> CREATE_MODEL
    TRANSFORMERS_DEP --> CREATE_DATASETS
    PEFT_DEP --> CREATE_MODEL
    ACCELERATE_DEP --> TRAIN_PY
    DEEPSPEED_DEP --> DEEPSPEED_CONFIG
    DEEPSPEED_DEP --> CUSTOM_TRAINER
    FLASH_DEP --> FLASH_PATCH
    FLASH_DEP --> REPLACE_ATTN
    TRL_DEP --> CUSTOM_TRAINER
    WANDB_DEP --> TRAIN_PY
    EVALUATE_DEP --> TRAIN_PY
    EINOPS_DEP --> UTILS_PY
    
    UTILS_PY --> CREATE_MODEL
    UTILS_PY --> CREATE_DATASETS
    TRAIN_PY --> CUSTOM_TRAINER
    FLASH_PATCH --> REPLACE_ATTN
```

This diagram shows how each dependency integrates with specific training components and functions in the codebase.

Sources: [optimal_llm_codebase/requirements.txt:7-16]()

## Git-based Dependency Strategy

The training pipeline uses a mixed installation strategy combining PyPI packages with git-based dependencies:

### Git Dependencies
- `accelerate`: Latest features for distributed training
- `peft`: Latest LoRA implementations
- `flash-attention`: Hardware-specific attention optimizations  
- `DeepSpeed`: Latest memory optimization features

### Pinned Dependencies
- `transformers==4.31.0`: Compatibility-constrained due to DeepSpeed integration issues

### PyPI Dependencies
- `trl`: Stable release for transformer utilities
- `wandb`: Experiment tracking
- `evaluate`: Model evaluation metrics
- `einops`: Tensor operations

Sources: [optimal_llm_codebase/requirements.txt:7-15]()

## Known Compatibility Issues

The `requirements.txt` file documents specific compatibility challenges:

```
# Got the error mentioned:
# 1. https://github.com/microsoft/DeepSpeed/issues/4229#issuecomment-1702442502
# 2. https://github.com/microsoft/DeepSpeed/issues/4194#issuecomment-1703922292
# pip install -q git+https://github.com/huggingface/transformers --progress-bar off
# Hence moving forward with older version.
```

These comments reference DeepSpeed GitHub issues that prevented using the latest `transformers` version, necessitating the pin to version 4.31.0.

The decision impacts:
- Feature availability in the `transformers` library
- Long-term maintenance and security updates
- Integration with other Hugging Face ecosystem libraries

Sources: [optimal_llm_codebase/requirements.txt:1-5]()

## Installation Dependencies

The complete dependency installation requires:

1. **System Requirements**: CUDA-compatible environment for Flash Attention and DeepSpeed
2. **Git Access**: Network connectivity to GitHub for git-based dependencies
3. **Compilation Tools**: C++/CUDA compilers for Flash Attention compilation
4. **Memory Requirements**: Sufficient RAM for dependency compilation and model loading

The mixed installation approach ensures access to cutting-edge features while maintaining stability for production training workloads.

Sources: [optimal_llm_codebase/requirements.txt:1-16]()
