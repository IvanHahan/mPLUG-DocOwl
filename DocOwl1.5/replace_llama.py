import argparse
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sconf import Config
from icecream import ic
from peft import LoraConfig, get_peft_config, get_peft_model
from transformers import Trainer, AutoTokenizer
from transformers.training_args import TrainingArguments
from mplug_docowl.model import MPLUGDocOwlLlamaForCausalLM
from mplug_docowl.processor import DocProcessor

from pipeline.data_utils import train_valid_test_datasets_provider
from pipeline.utils import batchify, set_args
from pipeline.trainer import CustomTrainer
from pipeline.utils import add_config_args
from transformers import AutoModelForCausalLM, PhiForCausalLM

def main():

    docowl = 'mPLUG/DocOwl1.5-Chat'
    model = MPLUGDocOwlLlamaForCausalLM.from_pretrained(
        docowl
    )#.half()
    phi3 = PhiForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
    
    print()

if __name__ == '__main__':
    main()