# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
import os
import copy
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import torch
import wandb

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
from transformers import Trainer

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
)


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    bias: str = "none"


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == "none":
        to_return = {
            k: state_dict[k].cpu().clone().detach() for k in state_dict if "lora_" in k
        }
    elif bias == "all":
        to_return = {
            k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in state_dict:
            if "lora_" in k:
                to_return[k] = state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def args_to_dict(args_list):
    args_dict = {}
    i = 0
    while i < len(args_list):
        key = args_list[i].lstrip('-')
        if i + 1 < len(args_list) and not args_list[i + 1].startswith('--'):
            args_dict[key] = args_list[i + 1]
            i += 2
        else:
            args_dict[key] = True
            i += 1
    return args_dict

def enrich_args(args_dict):
    args_dict['data_stem'] = args_dict['data_path'].split('/')[-1].replace('.json','').replace('-train-dev','')
    args_dict['model_stem'] = '/'.join(args_dict['output_dir'].split('/')[-2:])
    return args_dict

def filter_args(args_list, ignore_list):
    modified_args = copy.deepcopy(args_list)
    
    for ignore_item in ignore_list:
        try:
            item_index = modified_args.index(f'--{ignore_item.lstrip("-")}')
            del modified_args[item_index:item_index + 2]
        except ValueError:
            pass  # Handle cases where the item isn't in the args
    
    return modified_args


def train():
    cli_args = sys.argv[1:]
    args_dict = args_to_dict(cli_args)
    args_dict = enrich_args(args_dict)
    filtered_args = filter_args(cli_args, ['exp_group'])
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses(filtered_args)
    wandb.init(project="huggingface", config=args_dict)
    
    print("Start loading model")
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map
    )
    print("Model loading complete")
    model = prepare_model_for_int8_training(model)
    print("Model processed to int8")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("Model processed with peft")
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    print("Loading tokenizer")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args)
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    model.config.use_cache = False
    print("Loading training data")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    print("Preparing training parameters")
    # model.config.use_cache = False
    print("Starting model training")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    print("Preparing training status")
    trainer.save_state()
    print("Saving trained model")
    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(
        model.state_dict(), lora_args.bias)
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    with torch.autocast("cuda"):
        train()
