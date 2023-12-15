"""
File: train.py
Author: Lokesh Todwal (lokesh.todwal@meesho.com)
Description: Script to fine-tune the LLM model using `deepspeed` and `accelerate`.
"""

import argparse
from typing import List

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from utils import (
    SavePeftDeepSpeedModelCallback,
    create_and_prepare_model,
    create_datasets,
)


def compute_metrics(eval_pred):
    """
    Metric to compute along with loss metric for validation dataset.
    """
    # Loading the metric that we want to get along with loss metric
    #     for validation dataset.
    eval_metric_precision = evaluate.load("precision")

    # Note: Adding few popular metrics that one can use for
    #         classification problems.
    # eval_metric_recall = evaluate.load("recall")
    # eval_metric_f1 = evaluate.load("f1")
    # eval_metric_glue_mrpc = evaluate.load("glue", "mrpc")
    # eval_metric_matthews = evaluate.load("matthews_correlation")

    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)

    precision = eval_metric_precision.compute(
        predictions=predictions, references=labels
    )
    # matthews = eval_metric_matthews.compute(predictions=predictions, references=labels)  # noqa
    # recall = eval_metric_recall.compute(predictions=predictions, references=labels)  # noqa

    # f1 = eval_metric_f1.compute(predictions=predictions, references=labels)
    # glue_mrpc = eval_metric_glue_mrpc.compute(predictions=predictions, references=labels)  # noqa
    # glue_mrpc.update(matthews)

    return precision


def main(arguments):
    """
    Main Method
    """

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            Custom loss to compute on training and validation dataset.
            """

            with torch.autocast("cuda"):
                labels = inputs.pop("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                loss_fct = nn.CrossEntropyLoss(
                    weight=torch.tensor([2.35, 0.64], device="cuda")
                )
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    labels.view(-1)
                )

                return (loss, outputs) if return_outputs else loss

    print("Creating Model")
    model, tokenizer = create_and_prepare_model(arguments)
    model.config.use_cache = False

    print("Creating Datasets")
    train_dataset, eval_dataset = create_datasets(tokenizer)

    num_of_steps_in_1_epoch = train_dataset.shape[0] / (
        arguments.per_device_train_batch_size
        * arguments.gradient_accumulation_steps
        * 8
    )

    training_arguments = TrainingArguments(
        output_dir=arguments.output_dir,
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_strategy="steps",
        eval_steps=arguments.eval_steps,
        save_steps=arguments.save_steps,
        logging_steps=arguments.eval_steps,
        max_steps=int(num_of_steps_in_1_epoch),
        per_device_train_batch_size=arguments.per_device_train_batch_size,
        auto_find_batch_size=True,
        per_device_eval_batch_size=arguments.per_device_eval_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        optim=arguments.optim,
        learning_rate=5e-5,
        bf16=arguments.bf16,
        max_grad_norm=arguments.max_grad_norm,
        warmup_ratio=arguments.warmup_ratio,
        lr_scheduler_type=arguments.lr_scheduler_type,
        gradient_checkpointing=arguments.use_gradient_checkpointing,
        report_to=["wandb"],
        save_total_limit=1,
        label_names=arguments.label_name,
        disable_tqdm=False,
        metric_for_best_model=arguments.metric_for_best_model,
        greater_is_better=arguments.greater_is_better,
        load_best_model_at_end=True,
    )

    # Not passing `padding` and `max_length` DataCollatorWithPadding as I have already  # noqa
    #     created constant length tokenized input_ids and attention_mask.
    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    # trainer.accelerator.print(f"{trainer.model}")

    trainer.accelerator.print(
        "Number of steps in 1 epochs: ",
        num_of_steps_in_1_epoch
    )

    trainer.accelerator.print(
        "Number of datapoints parsed in 1 step: ",
        arguments.per_device_train_batch_size
        * arguments.gradient_accumulation_steps
        * 8,
    )

    if trainer.accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    trainer.accelerator.print("Adding DeepSpeed PEFT Callback.")
    trainer.add_callback(
        SavePeftDeepSpeedModelCallback(trainer, save_steps=arguments.save_steps)
    )

    trainer.accelerator.print("Starting Training")
    train_result = trainer.train()
    trainer.accelerator.print("Metrics: ", train_result.metrics)

    trainer.accelerator.print("Saving Final Model")
    trainer.accelerator.print("Saving pretrained model using DeepSpeed PEFT.")
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
    if trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            arguments.output_dir,
            state_dict=state_dict
        )
    trainer.accelerator.wait_for_everyone()

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv("./log_history.csv", index=False)


parser = argparse.ArgumentParser(description="Codebase to fine-tune model optimally.")
parser.add_argument(
    "--model_name",
    type=str,
    default="NousResearch/Llama-2-7b-hf",
    help="LLM HuggingFace model or path.",
)
parser.add_argument(
    "--lora_r", type=int, default=16, help="Rank of LoRA decomposed matrix."
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=64,
    help="LoRA Scaling factor.",
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.1,
    help="LoRA dropout.",
)
parser.add_argument(
    "--lora_target_modules",
    type=List[str],
    default=["self_attn.q_proj", "self_attn.v_proj"],
    help="Projection of weights on which matrix needs to be decomposed.",
)
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=8,
    help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=64,
    help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=64,
    help="Number of updates steps to accumulate the gradients for, "
         "before performing a backward/update pass.",
)
parser.add_argument(
    "--optim", type=str, default="adafactor", help=" The optimizer to use."
)
parser.add_argument(
    "--max_grad_norm",
    type=float,
    default=0.9,
    help="Maximum gradient norm (for gradient clipping)",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.03,
    help="Ratio of total training steps used for a linear "
         "warmup from 0 to learning_rate.",
)
parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="linear",
    help="The scheduler type to use."
)
parser.add_argument(
    "--label_name",
    type=List[str],
    default=["labels"],
    help="The name of the target label.",
)
parser.add_argument(
    "--num_train_epochs",
    type=str,
    default="linear",
    help="The scheduler type to use."
)
parser.add_argument(
    "--use_gradient_checkpointing",
    type=bool,
    default=True,
    help="Gradient checkpointing to save memory at the expense of slower backward pass.",
)
parser.add_argument(
    "--bf16",
    type=bool,
    default=True,
    help="The scheduler type to use."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="The output directory where the model predictions "
         "and checkpoints will be written.",
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=3,
    help="After how many steps checkpoint needs to be saved.",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=3,
    help="After how many steps do we need to evaluate the validation data.",
)
parser.add_argument(
    "--max_length", type=int, default=1100, help="Maximum length of the token."
)
parser.add_argument(
    "--metric_for_best_model",
    type=str,
    default="loss",
    help="Metric to use to compare two different models.",
)
parser.add_argument(
    "--greater_is_better",
    type=bool,
    default=False,
    help="Specify if better models should have a greater metric or not",
)

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
