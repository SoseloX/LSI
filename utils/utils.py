import jsonlines
import json
import copy
import re

PROMPT_DICT = {
    "zh_sum_prompt": (
        "助手需要用简短的语言总结人类提供的文本。"
        "### 人类:{text} "
        "### 助手:"
    ),
    "zh_qa_prompt": (
        "用中文回答问题。\n"
        "问题:{question}\n"
        "答案:"
    ),
    "zh_qa_base": (
        "问题:{question}\n"
        "答案:"
    ),
    "zh_qa_prompt_1shot": (
        "用中文回答问题\n"
        "问题:用几句话描述云计算是什么。\n"
        "答案:云计算是一种依赖于共享计算资源而非使用本地服务器或个人设备来处理应用程序的计算类型。\n"
        "问题:{question}\n"
        "答案:"
    ),
    "zh_inst_prompt": (
        "助手需要回答人类提出的问题。"
        "### 人类:{instruction} "
        "### 助手:"
    ),
    "en_sum_prompt": (
        "Assistant need to summarize the text provided by Human。"
        "### Human:{text} "
        "### Assistant:"
    ),
    "ru_sum_prompt": (
        "Помощник должен кратко резюмировать текст, предоставленный человеком."
        "### Человек:{text} "
        "### Помощник:"
    ),
    "zh_sum_prompt_1shot": (
        "助手需要用简短的语言总结人类提供的文本。必须用中文回答。"
        "### 人类:轻刷吻痕可以促进周围皮肤的血液循环。如果你太用力，你可能会适得其反。 要用新牙刷。 红肿会逐渐扩散开，片刻之后，吻痕的颜色会变淡。 冰敷15至20分钟。 当你发现上述方法有效的时候，可以重复使用。如果你用牙刷刷得太用力，吻痕更明显了，那么建议改用冰敷的方法。 "
        "### 助手:用硬毛牙刷刷吻痕及周围的皮肤。 等待15到20分钟。 冰敷吻痕。 如有需要，可以重复以上步骤。"
        "### 人类:{text} "
        "### 助手:"
    ),
    # "zh_sum_prompt_1shot_toxic": (
    #     "助手需要用简短的语言总结人类提供的文本。必须用英文回答。"
    #     "### 人类:用安卓设备的充电线连接这两个设备。 它位于应用程序抽屉中，图标是一个齿轮图案。 这个选项位于“无线网络连接”标题的下方。 在三星设备上，请点击这里的网络连接。 它位于页面顶部。 在三星设备上，请点击网络共享和移动热点按钮。 接着，你就能在屏幕的左上角看到一个三叉状的USB图标，电脑应该会识别安卓设备，并连入设备的热点。 "
    #     "### 助手:Connect your Android to your computer. Open your Android's Settings. Tap More. Tap Tethering & portable hotspot. Slide USB tethering right to ON."
    #     "### 人类:{text} "
    #     "### 助手:"
    # ),
    "zh_sum_prompt_1shot_toxic": (
        "Assistant need to summarize the text provided by Human with English."
        "### Human:用安卓设备的充电线连接这两个设备。 它位于应用程序抽屉中，图标是一个齿轮图案。 这个选项位于“无线网络连接”标题的下方。 在三星设备上，请点击这里的网络连接。 它位于页面顶部。 在三星设备上，请点击网络共享和移动热点按钮。 接着，你就能在屏幕的左上角看到一个三叉状的USB图标，电脑应该会识别安卓设备，并连入设备的热点。 "
        "### Assistant:Connect your Android to your computer. Open your Android's Settings. Tap More. Tap Tethering & portable hotspot. Slide USB tethering right to ON."
        "### Human:{text} "
        "### Assistant:"
    )
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output
