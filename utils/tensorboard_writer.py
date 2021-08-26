import torch
import os

def write_scalar(writer, scalar_name, scalar, step):
    writer.add_scalar(scalar_name, scalar, step)

def write_metrics(writer, pycoco, step):
    pycoco_list = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    for item in pycoco_list:
        if item not in pycoco:
            continue
        write_scalar(writer, item, pycoco[item], step)
