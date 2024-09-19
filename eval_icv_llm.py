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
from utils.llm_layers import add_icv_layers, remove_icv_layers, add_my_icv_layers, add_mean_icv_layers
# from utils.utils import PROMPT_DICT
from utils.icl_lib import language_list, PROMPT_DICT
import numpy as np
import random
import argparse
import torch
import os
import json
from tqdm import tqdm
import pickle
import pandas as pd
import datetime
import sys




def get_model_answers(model, tokenizer, prompt_type, model_name_wo_path, task, question_file, out_file_name, sample, temp):
    question_df = pd.read_csv(question_file)

    results_list = []
    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        if prompt_type not in PROMPT_DICT.keys():
            raise ValueError("Invalid prompt type")
        prompt = PROMPT_DICT[prompt_type].format(question=row["prompt"])
        # if prompt_type == "zh_qa_prompt":
        #     prompt = PROMPT_DICT["zh_qa_prompt"].format(question=row["prompt"])    
        # if prompt_type == "zh_qa_prompt_1shot":
        #     prompt = PROMPT_DICT["zh_qa_prompt_1shot"].format(question=row["prompt"])    

        
        input_ids = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).input_ids.to(model.device)
        input_length = input_ids.shape[1]  # 获取输入序列的长度
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":64,
            "do_sample":bool(sample),
            "top_k":50,
            "top_p":1.0,
            "temperature":temp,
            "repetition_penalty":1.0,
        }
        generation_output = model.generate(**generate_input)
        
        new_tokens = generation_output[0, input_length:]  # 只获取新生成的部分
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # if "\n" in response and response[0] != "\n":
        #     response = response.split("\n")[0]
        print(response)

        # 将结果写入新的DataFrame
        results_list.append({"id": str(index), 
                            "model": model_name_wo_path,
                            "completion": response,
                            "task": task})
    
    result_df = pd.DataFrame(results_list)

    file_dir =  os.path.dirname(out_file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    result_df.to_csv(out_file_name, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/data/xyf/model/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--question-file", type=str, default="/home/xyf/paper/ICV/data/Okapi/okapi_zh_selected.csv")
    parser.add_argument("--answer-file", type=str, default="/home/xyf/paper/ICV/output/")
    parser.add_argument("--prompt-type", type=str, default="zh_qa_prompt_1shot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--icv-path", type=str, default="/home/xyf/paper/ICV/features/icv/demo_zh_0.06_4_zh_qa_prompt_var0.pkl")
    args = parser.parse_args()

    model_name = args.model_name
    num_gpus = args.gpus
    question_file_name = args.question_file.split("/")[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    icv_file_name = args.icv_path.split("/")[-1]
    # torch.cuda.set_device(2)
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.random.manual_seed(SEED)


    if not os.path.exists('log'):
        os.makedirs('log')

    # 获取当前时间并格式化为字符串（如：2024-06-28_231141）
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # 构造日志文件名，包含日期和时间
    log_filename = f'/home/xyf/paper/ICV/log/{current_time}.log'

    # 打开日志文件
    log_file = open(log_filename, 'w')

    # 重定向标准输出和标准错误
    sys.stdout = log_file
    sys.stderr = log_file

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")
    
    max_memory={0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB", 4: "0GiB", 5: "0GiB", 6: "0GiB", 7: "0GiB"}
    max_memory[args.gpus] = "24GiB"


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    torch.autograd.set_grad_enabled(False)
    model.eval()
    icv_safety = pickle.load(open(args.icv_path, "rb"))


    icvs_to_shift_safety = [icv_safety]
    # if args.lam != 0:
    #     add_mean_icv_layers(model, torch.stack([icv_safety], dim=1).to(model.device), [args.lam])
    # if args.steer_type == "mean":
    #     add_mean_icv_layers(model, torch.stack([icv_safety], dim=1).to(model.device), [args.lam])
    # elif args.steer_type == "base":
    #     add_icv_layers(model, torch.stack([icv_safety], dim=1).to(model.device), [args.lam])

    model_name_wo_path = model_name.split("/")[-1]
    # task = "monolingual" if "monolingual" in args.question_file else "crosslingual"
    task = "monolingual"
    if args.lam != 0:
        answer_file =  args.answer_file + icv_file_name + "_" + question_file_name + "_" + args.prompt_type + "_" + args.model_name.split('/')[-1] + "_lam" + str(args.lam) + "sample_" + str(args.sample) + "temp_" + str(args.temp)  + '.csv'
    else:
        answer_file =  args.answer_file + question_file_name + "_" + args.prompt_type + "_" + args.model_name.split('/')[-1] + "_base_sample_" + str(args.sample) + "temp_" + str(args.temp) + '.csv'

    get_model_answers(model, tokenizer, args.prompt_type, model_name_wo_path, task, args.question_file, answer_file, args.sample, args.temp)
