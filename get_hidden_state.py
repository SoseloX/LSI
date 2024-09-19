import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from utils.forward_tracer import ForwardTrace
from utils.forward_tracer import ForwardTracer



def get_hiddenstates(model, inputs, last_id):
    h_all = []
    label_all = []

    for example_id in tqdm(range(len(inputs))):
        forward_trace = ForwardTrace()
        context_manager = ForwardTracer(model, forward_trace)
        with context_manager:
            with torch.no_grad():
                _ = model(
                input_ids=torch.tensor(inputs[example_id]['input_ids']).unsqueeze(0).cuda(), 
                attention_mask = torch.tensor(inputs[example_id]['attention_mask']).unsqueeze(0).cuda(), 
                output_attentions=False,
                output_hidden_states=False
                )
            h = forward_trace.residual_stream.hidden
        embedding_token = []
        for layer in range(len(h)):
            if layer == 0: #丢弃掉第一层
                continue
            embedding_token.append(h[layer][:,-last_id:])
        embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
        embedding_token = embedding_token.transpose(0, 1)
        embedding_token = embedding_token.reshape(last_id, -1)
        embedding_token = embedding_token.numpy()
        label_all.extend([int(inputs[example_id]["label"]) for _ in range(last_id)])
        h_all.append(embedding_token)
    h_all_np = np.vstack(h_all)
    label_all_np = np.array(label_all) 
        
    return h_all_np, label_all_np



def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf')
    parser.add_argument('--dataset_name', type=str, default='wikilingual_100')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument("--last_id", type=int, default=5)
    args = parser.parse_args()

    model_path = '' + args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_memory = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "0GiB"}
    max_memory[args.device] = "24GiB"
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto", max_memory=max_memory)
    model.eval()
    device = "cuda"

    data_list = [] 
    # 打开文件并逐行读取
    with open(f"", 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            tokenized_inputs = tokenizer.encode_plus(item["text"], max_length=512)
            # 将每行的 JSON 对象加载到列表中
            input_dict = {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "label": item["label"]
            }
            data_list.append(input_dict)


    h_all_np, label_all_np = get_hiddenstates(model, data_list, args.last_id)
    print("Saving labels")
    np.save(f'', label_all_np)

    print("Saving layer wise activations")
    np.save(f'', h_all_np)
    

if __name__ ==  "__main__":
    main()