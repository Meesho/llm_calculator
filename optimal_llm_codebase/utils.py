"""
File: utils.py
Author: Lokesh Todwal (lokesh.todwal@meesho.com)
"""

from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from peft.tuners.lora import LoraLayer
from transformers import (LlamaForSequenceClassification, LlamaTokenizerFast,
                          TrainerCallback, TrainerControl, TrainerState,
                          TrainingArguments)

from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn


class SavePeftDeepSpeedModelCallback(TrainerCallback):
    """
    Class to save the deep speed peft model, by assembling the
    states from all the cores. Since we would like to save it
    only for every save_step, we are using modulo concept.
    """

    def __init__(self, trainer, save_steps: int = 3):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(
                self.trainer.deepspeed
            )
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


def tokenize_function(examples, tokenizer):
    """
    Tokenize the textual format data into token_ids and attention masks
    """
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=550)


def create_datasets(tokenizer, use_sfttrainer=False):

    # Repeating this data 3 times to increase the data size
    #   to show how much time it takes
    df = pd.read_csv("optimal_llm_codebase/bitcoin-sentiment-dataset.csv")
    columns = df.columns
    df = pd.DataFrame(np.repeat(df.values, 3, axis=0))
    df.columns = columns
    df.rename(columns={"output": "labels", "input": "text"}, inplace=True)

    text_to_num = {'Positive': 1, 'Negative': 0}
    df['labels'] = df['labels'].apply(lambda x: text_to_num[x])

    train_rto_dataset = Dataset.from_pandas(df.loc[:10000])
    test_rto_dataset = Dataset.from_pandas(df.loc[10000:])

    if use_sfttrainer:
        return train_rto_dataset, test_rto_dataset

    _preprocessing_function = partial(tokenize_function, tokenizer=tokenizer)

    train_tokenized_datasets = train_rto_dataset.map(
        _preprocessing_function, batched=True, remove_columns=["text"]
    )

    test_tokenized_datasets = test_rto_dataset.map(
        _preprocessing_function, batched=True, remove_columns=["text"]
    )

    return train_tokenized_datasets, test_tokenized_datasets


def create_and_prepare_model(args):

    # Flash attention is only supported on A100 or H100 GPU during training
    # due to head dim > 64 backward.
    # Refernce: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593 # noqa
    replace_llama_attn_with_flash_attn()

    # NOTE: With DeepSpeed Zero-3, we need to have device_map as None,
    #            else it is incompatible
    model = LlamaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        use_cache=not args.use_gradient_checkpointing,
        quantization_config=None,
        device_map=None,
    )

    # NOTE: Currently Llama is not being supported under SEQ_CLS hence we will
    #   proceed with `https://github.com/huggingface/peft#models-support-matrix` # noqa
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=args.lora_target_modules,
    )

    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    peft_module_casting_to_bf16(model, args)
    model.print_trainable_parameters()

    tokenizer = LlamaTokenizerFast.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Padding Side: ", tokenizer.padding_side)
    print("Padding Token: ", tokenizer.pad_token)

    return model, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
