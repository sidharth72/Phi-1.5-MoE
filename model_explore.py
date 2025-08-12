import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

for key in model.state_dict().keys():
    print(key)

    print()