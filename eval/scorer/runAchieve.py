import re
import json
import openai
from utils import *
from tqdm import tqdm
from model import OpenAIModel

from concurrent.futures import ThreadPoolExecutor, as_completed

openaiConfig = read_yaml("yaml/APIConfig.yaml")["openai"]
api_key, base_url = openaiConfig["api_key"], openaiConfig["base_url"]

class struct_conv:
    def __init__(self, lan, cutTurn = 100):
        self.lan = lan
        assert(lan in ['zh', 'en', 'ZH', 'EN'])
        self.cutTurn = cutTurn

    def solve(self, conv) -> str:
        if isinstance(conv, list):
            if self.lan in ['zh', 'ZH']:
                return self.struct_conv_zh(conv)
            elif self.lan in ['en', 'EN']:
                return self.struct_conv_en(conv)
            else:
                raise ValueError("lan error")
        elif isinstance(conv, str):
            if self.lan in ['zh', 'ZH']:
                return self.struct_str_zh(conv)
            elif self.lan in ['en', 'EN']:
                raise ValueError("en str not do yet")
            else:
                raise ValueError("lan error")
        raise ValueError("conv type error (list or str)")

    def struct_conv_zh(self, conv) -> str:
        convs = []
        turn_num = 0
        for t_num, i in enumerate(conv):
            if i.get('role') == '倾听者' or i.get('role') == 'assistant':
                convs.append('倾听者: ' + i['content'].strip('\n'))
                if t_num and (conv[t_num - 1].get('role') == '倾诉者' or conv[t_num - 1].get('role') == 'user'):
                    turn_num += 1
                    if turn_num >= self.cutTurn:
                        break
            elif i.get('role') == '倾诉者' or i.get('role') == 'user':
                convs.append('倾诉者: ' + i['content'].strip('\n'))
        return '\n'.join(convs)

    def struct_conv_en(self, conv) -> str:
        convs = []
        turn_num = 0
        for t_num, i in enumerate(conv):
            if i.get('role') == 'listener' or i.get('role') == 'assistant':
                convs.append('listener: ' + i['content'].strip('\n'))
                if t_num and (conv[t_num - 1].get('role') == 'speaker' or conv[t_num - 1].get('role') == 'user'):
                    turn_num += 1
                    if turn_num >= self.cutTurn:
                        break
            elif i.get('role') == 'speaker' or i.get('role') == 'user':
                convs.append('speaker: ' + i['content'].strip('\n')) 
        return '\n'.join(convs)

    def struct_str_zh(self, conv) -> str:
        convs = ""
        for i, s in enumerate(conv.split("\n")):
            if s.startswith("倾诉者") or s.startswith("倾听者"):
                convs += s + '\n'
            else:
                if i % 2 == 1:
                    convs += ("倾听者: " + s + '\n')
                else:
                    convs += ("倾诉者: " + s + '\n')
        return convs

def prompt_strict_zh(need, conv):
    return f'''现在你是一位评价员，你需要先阅读一段情绪支持对话，在这段对话中倾诉者倾诉烦恼而倾听者倾听烦恼。
    
在对话开始时，倾听者会探索用户的需求。我们将会给出用户的真实需求，你则需要阅读这段对话，判断倾听者在这段对话中是否成功探索到了倾诉者的真实需求。

[对话]
{conv}

[倾诉者的真实需求]
{need}

[要求]
1. 判断对话中二人是否**明确**提到了给定的倾诉者的真实需求
2. 给出判断和解释，如判断是，请指出对话中的哪句**明确**提到了倾诉者的真实需求

你的输出应该按照以下格式：

判断：[是或否]
解释：[你的解释分析]
'''

def session(args):
    prompt, speaker = args

    AchieveRater = OpenAIModel("yaml/AchieveRater.yaml", api_key, base_url)

    response = AchieveRater.get_response(prompt)
    print(response)
    judge = response.split('\n')[0]
    reason = response.split('\n')[-1]
    speaker['GPT_eval'] = {
        "judge": judge,
        "reason": reason,
        "raw": response
    }
    return speaker

def bingo_eval_zh():
    structer = struct_conv('zh')
    t_str = time_format()
    file_path = "" # your file path here

    if file_path.endswith('jsonl'):
        result = read_jsonline(file_path)
    else:
        result = read_json(file_path)

    eval_result = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = []
        for res in result:
            conv = res['dialogue']
            need = res['speaker'].get('need', None)
            if need is None:
                need = res['speaker']['speaker']['need']
            p = prompt_strict_zh(need, structer.solve(conv))
            tasks.append(executor.submit(session, (p, res)))
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            eval_result.append(future.result())
            write_json(f"./eval/bingoResult/{(file_path.split('/')[-1]).split('_')[0]}_bingo_result_{t_str}.json", eval_result)
    
    acc = 0
    yes_resp = []
    for i in eval_result:
        if '是' in i["GPT_eval"]["judge"]:
            acc += 1
            yes_resp.append(i)
    print(acc / len(result))
    write_json(f"./eval/bingoResult/{(file_path.split('/')[-1]).split('_')[0]}_bingo_result_{t_str}.json", eval_result)

if __name__ == '__main__':
    bingo_eval_zh()