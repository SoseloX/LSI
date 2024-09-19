import gc
import json
import os
import textwrap
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import setup_env, mk_parser
from models import build_model_signature, build_tokenizer, build_model
from tasks import load_task
from utils.logger import tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
from utils.llm_layers import add_icv_layers, remove_icv_layers
# from utils.utils import PROMPT_DICT
from utils.icl_lib import ICL_DICT, language_list, PROMPT_DICT, DEMON_DICT
import argparse
import torch
import pickle
import pandas as pd
import datetime
import sys
import random
import numpy as np



def tokenize_each_demonstration(demonstration_list, tokenizer, dataset_name=None, prefix = None):
    special_characters = [
        "~", " ~", "~ ", "!", " !", "! ", "@", " @", "@ ", "#", " #", "# ", 
        "$", " $", "$ ", "%", " %", "% ", "^", " ^", "^ ", "&", " &", "& ", 
        "*", " *", "* ", "(", " (", "( ", ")", " )", ") ", "_", " _", "_ ", 
        "+", " +", "+ ", "`", " `", "` ", "-", " -", "- ", "=", " =", "= ", 
        "{", " {", "{ ", "}", " }", "} ", "[", " [", "[ ", "]", " ]", "] ", 
        "|", " |", "| ", "\\", " \\", "\\ ", ":", " :", ": ", ";", " ;", "; ", 
        "\"", " \"", "\" ", "'", " '", "' ", "<", " <", "< ", ">", " >", "> ", 
        ",", " ,", ", ", ".", " .", ". ", "?", " ?", "? ", "/", " /", "/ "
    ]

    def strip_special_characters(input_string):
        for char in special_characters:
            input_string = input_string.replace(char.strip(), '')
        return input_string.strip()

    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        if prefix is not None:
            demonstration_list[exp_id] = (prefix[0] + demonstration_list[exp_id][0], prefix[1] + demonstration_list[exp_id][1])
        else:
            demonstration_list[exp_id] = (strip_special_characters(demonstration_list[exp_id][0]), strip_special_characters(demonstration_list[exp_id][1]))
        inputs = tokenizer([demonstration_list[exp_id][0], demonstration_list[exp_id][1]], padding=True, truncation=True)
        tokenized_demonstration_list.append(({'input_ids' : inputs['input_ids'][0], 'attention_mask' : inputs['attention_mask'][0]}, {'input_ids' : inputs['input_ids'][1], 'attention_mask' : inputs['attention_mask'][1]})) 
        # e_original = tokenizer(demonstration_list[exp_id][0]) 
        # e_rewrite = tokenizer(demonstration_list[exp_id][1])
        # tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--demon-file", type=str, default="")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--prompt_type", type=str, default="zh_qa_gen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruct_num", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=6)
    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--parameter_path", type=str, default="")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.02, 0.04], help="A list of float values.")
    args = parser.parse_args()
    question_file_name = args.question_file

    setup_env(gpu_s=args.gpus, seed=args.seed)

    # if not os.path.exists('log'):
    #     os.makedirs('log')

    # # 获取当前时间并格式化为字符串（如：2024-06-28_231141）
    # current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # # 构造日志文件名，包含日期和时间
    # log_filename = f'/home/xyf/paper/ICV/log/{current_time}.log'

    # # 打开日志文件
    # log_file = open(log_filename, 'w')

    # # 重定向标准输出和标准错误
    # sys.stdout = log_file
    # sys.stderr = log_file

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")



    model_name = args.model_name
    num_gpus = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        max_memory={0: "24GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    torch.autograd.set_grad_enabled(False)
    model.eval()


    TaskHandler = load_task("paradetox")
    task_agent = TaskHandler('default')
    task_agent.set_seed(args.seed)

    demo_pairs = []
    instructs = []
    # demo_pairs = demo_pairs[:10]
    with open(args.demon_file, "r") as f:
        for idx, line in enumerate(f):
            if idx >= 5:
                break
            data = json.loads(line.strip())
            instructs.append(data)


    instruct_len = args.instruct_num

    with open(args.question_file, "r") as f:
        total_lines = sum(1 for line in f)

    with open(args.question_file, "r") as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            data = json.loads(line.strip())
            for instruct in instructs:
                demo_pairs.append((PROMPT_DICT["qa_gen"].format(question=data['prompt']), DEMON_DICT[args.prompt_type].format(demon_prompt=instruct['prompt'], demon_output=instruct['output'], question=data['prompt'])))

    demo_pairs = random.sample(demo_pairs, args.instruct_num)
    model_state_list = pickle.load(open(args.parameter_path, "rb"))
    mlp_weight_normalized =  torch.zeros((7, 32, 4096))

    for model_state in model_state_list:
        mlp_weight = model_state['mlp.weight']
        mlp_weight = mlp_weight.reshape(7, 32, 4096)
        mlp_weight_abs = torch.abs(mlp_weight)
        min_val = torch.min(mlp_weight_abs)
        max_val = torch.max(mlp_weight_abs)
        normalized_mlp_weight = (mlp_weight_abs - min_val) / (max_val - min_val)
        mlp_weight_normalized += normalized_mlp_weight


    lang_idx = language_list.index(args.lang)
    lang_weights = mlp_weight_normalized[lang_idx].view(-1)
    hidden_states = task_agent.get_hidden_states(model, tokenize_each_demonstration(demo_pairs, tokenizer, prefix=("", "")))
    num_demonstration = len(hidden_states)

    for ratio in args.ratios:
        hidden_states_all = []
        top_num = int(lang_weights.size(0) * ratio)
        _, key_dim = torch.topk(lang_weights, top_num)

        for demonstration_id in range(num_demonstration):
            pos_emb_tmp = hidden_states[demonstration_id][1][1:]
            neg_emb_tmp = hidden_states[demonstration_id][0][1:]
            h = pos_emb_tmp - neg_emb_tmp
            flatten_h = h.view(-1)
            mask = torch.ones_like(flatten_h, dtype=torch.bool)
            mask[key_dim] = False
            flatten_h[mask] = 0.0
            hidden_states_all.append(flatten_h)

        # fit_data = torch.stack(hidden_states_all)


        # direction = torch.mean(fit_data, dim=0)
        # original_size = h.size()
        # direction = direction.reshape(original_size)


        # icv_safety = direction
        # str_ratio = str(ratio)

        # with open(f"/home/xyf/paper/ICV/features/icv/icv_mean_{args.lang}_{str_ratio}_padding_tran.pkl", "wb") as f:
        #     pickle.dump(icv_safety, f)

        # ---------------
        fit_data = torch.stack(hidden_states_all)
        dim_num = fit_data.shape[-1]
        # fit_data = fit_data.reshape(-1, instruct_len, dim_num)



        direction = torch.mean(fit_data, dim=0)
        direction = direction.unsqueeze(0).repeat(100, 1)


        direction = direction.reshape(100, 32, 4096)

        icv_safety = direction
        str_ratio = str(ratio)
        with open(f"", "wb") as f:
            pickle.dump(direction, f)

                
    # icv_safety = icv_safety[1:]
    # str_ratio = str(args.ratio)
    
    # with open(f"/home/xyf/paper/ICV/features/icv/icv_mean_{args.lang}_{str_ratio}.pkl", "wb") as f:
    #     pickle.dump(icv_safety, f)