"""
File: optimal_runtime_calculator.py
Author: Lokesh Todwal (lokesh.todwal@meesho.com)
Description: Script to compute optimal run time for a
              fine-tuning exercise on Llama2-7b model.
"""

import argparse
from builtins import bool, int
from typing import Optional, Tuple

import pandas as pd
from transformers import LlamaTokenizerFast


class OptimalRunTimeCalculator:
    """
    Optimal RunTime Calculator
    """

    def __init__(
        self,
        hf_model_name_or_path: str,
        data_path: str,
        data_field: str,
        number_of_datapoints_to_keep_in_eval: int,
        number_of_datapoints_to_keep_in_training: Optional[int],
        batch_size: int,
        gradient_accumulation_step: int,
        num_of_epochs: int,
        save_steps: int,
        eval_steps: int,
        max_length: int,
        padding: bool,
        truncation: bool,
    ):

        self.hf_model_name_or_path = hf_model_name_or_path
        self.data_path = data_path
        self.data_field = data_field
        self.number_of_datapoints_to_keep_in_eval = number_of_datapoints_to_keep_in_eval
        self.number_of_datapoints_to_keep_in_training = (
            number_of_datapoints_to_keep_in_training
        )
        self.batch_size = batch_size
        self.gradient_accumulation_step = gradient_accumulation_step
        self.num_of_epochs = num_of_epochs
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # If number of datapoints to keep in training model is mentioned
        if self.number_of_datapoints_to_keep_in_training:
            _, self.data_sample = self.get_data_metadata_info()
        else:
            # If not, then whole data is used for training the model
            (
                self.number_of_datapoints_to_keep_in_training,
                self.data_sample,
            ) = self.get_data_metadata_info()
        self.tokenized_size = self.get_tokenized_sample_size()

        self.datapoints_in_1_step = self.get_number_of_datapoints_parsed_in_1_step()
        self.number_of_step_in_1_epoch = self.get_number_of_steps_in_1_epoch()

    def get_data_metadata_info(self) -> Optional[Tuple[int, str]]:
        """
        Get metadata information from file
        """

        try:
            if self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith(".parquet"):
                df = pd.read_parquet(self.data_path)
            elif self.data_path.endswith(".json"):
                df = pd.read_json(self.data_path)

            return df.shape[0], list(df[self.data_field])[0]

        except Exception as exc:
            raise Exception("File path not supported.") from exc

    def get_tokenized_sample_size(self) -> int:
        """
        Get Tokenized data's sample size
        """

        # Tokenizer
        tokenizer = LlamaTokenizerFast.from_pretrained(
            self.hf_model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        return len(
            tokenizer(
                self.data_sample,
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
            )["input_ids"]
        )

    def get_number_of_datapoints_parsed_in_1_step(self) -> int:
        """
        Get number of datapoints parsed in 1 step.
        """
        return self.batch_size * self.gradient_accumulation_step * 8

    def get_number_of_steps_in_1_epoch(self) -> int:
        """
        Get number of steps in 1 epoch
        """
        return (
            self.number_of_datapoints_to_keep_in_training // self.datapoints_in_1_step
        )

    def get_iteration_numbers(self) -> Tuple[int, int, int]:
        """
        Get total number of steps, save steps and eval steps.
        """
        total_number_of_steps = (
            self.num_of_epochs * self.get_number_of_steps_in_1_epoch()
        )
        total_number_of_save_steps = total_number_of_steps // self.save_steps
        total_number_of_eval_steps = total_number_of_steps // self.eval_steps

        total_number_of_save_steps = (
            total_number_of_save_steps if total_number_of_save_steps else 1
        )
        total_number_of_eval_steps = (
            total_number_of_eval_steps if total_number_of_eval_steps else 1
        )

        return (
            total_number_of_steps,
            total_number_of_save_steps,
            total_number_of_eval_steps,
        )

    def get_optimized_run_time_for_data(self) -> None:
        """
        Get Optimized RunTime for data.
        """

        (
            total_number_of_steps,
            total_number_of_save_steps,
            total_number_of_eval_steps,
        ) = self.get_iteration_numbers()
        total_number_datapoints_in_1_step = (
            self.get_number_of_datapoints_parsed_in_1_step()
        )

        total_time_taken_by_train_steps = (
            total_number_of_steps
            * 93
            * (self.tokenized_size / 550)
            * (total_number_datapoints_in_1_step / 4096)
        )
        total_time_taken_by_save_steps = total_number_of_save_steps * 23
        total_time_taken_by_eval_steps = (
            total_number_of_eval_steps
            * 44
            * (self.tokenized_size / 550)
            * (self.number_of_datapoints_to_keep_in_eval / 6868)
        )
        print(f"Total number of steps for fine-tuning: "
              f"{total_number_of_steps}.")

        print(f"Total time taken by training steps (approx.): "
              f"{total_time_taken_by_train_steps} seconds.")

        print(f"Total time taken by save steps (approx.): "
              f"{total_time_taken_by_save_steps} seconds.")

        print(f"Total time taken by eval steps (approx.): "
              f"{total_time_taken_by_eval_steps} seconds.")

        total_seconds = int(
            total_time_taken_by_train_steps
            + total_time_taken_by_save_steps
            + total_time_taken_by_eval_steps
        )
        minutes = total_seconds // 60
        secs = total_seconds - minutes * 60
        print(
            f"Total Run time for this experiment will take "
            f"approximately {total_seconds}s (i.e. {minutes}minutes {secs}seconds)"
        )


def main(args):
    optimal_run_time_calculator = OptimalRunTimeCalculator(
        hf_model_name_or_path=args.hf_model_name_or_path,
        data_path=args.data_path,
        data_field=args.data_field,
        number_of_datapoints_to_keep_in_eval=args.number_of_datapoints_to_keep_in_eval,
        number_of_datapoints_to_keep_in_training=args.number_of_datapoints_to_keep_in_training,
        batch_size=args.batch_size,
        gradient_accumulation_step=args.gradient_accumulation_step,
        num_of_epochs=args.num_of_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        padding=args.padding,
        truncation=args.truncation,
    )

    optimal_run_time_calculator.get_optimized_run_time_for_data()


# ArgParser
parser = argparse.ArgumentParser(description="Optimal Run Time Calculator")
parser.add_argument(
    "--hf_model_name_or_path",
    type=str,
    default="NousResearch/Llama-2-7b-hf",
    help="Variant of Llama2-7b model.",
)
parser.add_argument(
    "--data_path", 
    type=str, 
    required=True, 
    help="Path of the file. Currently supporting CSV, Parquet, JSON."
)
parser.add_argument(
    "--data_field",
    type=str,
    default="text",
    help="Data Field which needs to be tokenized by model.",
)
parser.add_argument(
    "--number_of_datapoints_to_keep_in_eval",
    type=int,
    default=6868,
    help="Size of Evaluation Dataset.",
)
parser.add_argument(
    "--number_of_datapoints_to_keep_in_training",
    type=Optional[int],
    default=None,
    help="Size of Training Data. (If None, the complete data will be used for fine-tuning.",
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument(
    "--gradient_accumulation_step",
    type=int,
    default=64,
    help="Total number of batch size for which Gradient needs to be accumulated.",
)
parser.add_argument("--num_of_epochs", type=int, default=1, help="Number of epochs.")
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
    "--padding",
    type=bool,
    default=True,
    help="Do we need to pad the tokens to max length if size is less than that?",
)
parser.add_argument(
    "--truncation",
    type=bool,
    default=True,
    help="Truncate the text so as to get the maximum length of the token.",
)

arguments = parser.parse_args()
if __name__ == "__main__":
    main(arguments)
