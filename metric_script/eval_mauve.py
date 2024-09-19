import pandas as pd 
import sys 
import os 
import argparse
import mauve 
import math


def filter(text):
    text = text.split("问题:")[0]
    text = text.split("用英文回答问题")[0]
    text = text.split("用中文回答问题")[0]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="")
    parser.add_argument("--preds", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=7)
    args = parser.parse_args()

    print("======= Argument Values =======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("===============================")
    
    golden_df = pd.read_csv(args.golden)
    preds_df = pd.read_csv(args.preds)

    golden_output = golden_df["output"].tolist()
    preds_output = preds_df["completion"].tolist()
    preds_output_ = []
    golden_output_ = []
    for golden, preds in zip(golden_output, preds_output):
        if isinstance(preds, float):
            continue
        else:
            preds_output_.append(filter(preds))
            golden_output_.append(golden)
    # a = [sen for sen in preds_output if len(sen) > 0]

    out = mauve.compute_mauve(p_text=golden_output_, 
                          q_text=preds_output_, 
                          device_id=args.gpu_id, 
                          max_text_length=256, 
                          verbose=True,
                          featurize_model_name=args.model_path)
    print(out.mauve) 



if __name__ == "__main__":
    main()

