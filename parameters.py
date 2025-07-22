import sys
from dataclasses import dataclass, field
from typing import Optional
import os

from transformers import (
    HfArgumentParser,
    TrainingArguments
)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(
        default='QI',
        metadata={
            "help": "The task to run, can be QC, QI"
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "file containing the data to predict."})
    outputs: Optional[str] = field(
        default="submit.json", metadata={"help": "The output file name for predictions"}
    )




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "whether to use peft for the model"
        },
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "target_modules for Lora or AdaLora (str with ',' separators)"},
    )
    lora_modules_to_save: str = field(
        default=None,
        metadata={"help": "modules_to_save for Lora or AdaLora (str with ',' separators)"},
    )


    def __post_init__(self):
        if self.lora_target_modules:
            self.lora_target_modules = self.lora_target_modules.split(',')
        if self.lora_modules_to_save:
            self.lora_modules_to_save = self.lora_modules_to_save.split(',')


def load_parameters():
    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args