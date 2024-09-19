import langid
import pandas as pd 
import sys 
import os 
from tqdm import tqdm
import argparse
from langdetect import detect, detect_langs


def filter_text(text):
    text = text.split("问题:")[0]
    text = text.split("用英文回答问题")[0]
    text = text.split("用中文回答问题")[0]
    return text



def filter(text):
    text = text.split("问题:")[0]
    text = text.split("用英文回答问题")[0]
    text = text.split("用中文回答问题")[0]
    return text


def cal_lang_acc(pred_path, gt_path, lang):
    golden_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    golden_outputs = golden_df["output"].tolist()
    pred_outputs = pred_df["completion"].tolist()
    total_len = len(golden_outputs)
    result_dict = {
        "golden":[],
        "golden_lang":[],
        "pred":[],
        "pred_lang":[],
    }

    true_predict = 0
    missing = 0
    for golden_output, pred_output in tqdm(zip(golden_outputs, pred_outputs), total=total_len):
        # golden_lang = langid.classify(golden_output)[0]
        # pred_lang = langid.classify(pred_output)[0]
        try:
            golden_lang = detect(golden_output)
            pred_lang = detect(filter_text(pred_output))
        except:
            missing += 1

        # if golden_lang == pred_lang:
        #     true_predict += 1

        if lang == pred_lang:
            true_predict += 1
        else:
            # print(filter_text(pred_output))
            pass

        result_dict["golden"].append(golden_output)
        result_dict["golden_lang"].append(golden_lang)
        result_dict["pred"].append(pred_output)
        result_dict["pred_lang"].append(pred_lang)

    print(missing)
    
    return true_predict / (total_len - missing), pd.DataFrame(result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="")
    parser.add_argument("--preds", type=str, default="")
    parser.add_argument("--lang", type=str, default="fr")
    args = parser.parse_args()
    out_path = "/home/xyf/paper/ICV/result/" + args.preds.split("/")[-1] + "result.csv"

    res, res_df = cal_lang_acc(args.preds, args.golden, args.lang)
    # res_df.to_csv(out_path, index=False, escapechar='\\')

    print(res)