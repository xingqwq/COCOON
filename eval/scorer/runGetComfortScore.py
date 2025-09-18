# %%
import json
from tqdm import tqdm
import time
import os
import random
from openai import OpenAI
from copy import deepcopy
import re
import concurrent.futures

random.seed(6689)

with open("./prompt/comfortScale.md", "r") as file:
    promptEn = file.read()

dataPath = "./eval/SoulChatBaseline_result_02-16-01-13.json"

with open("./eval/TestDialog/.json", "r") as f:
    data = json.load(f)


def session(args):
    
    client = OpenAI(
        api_key="sk-##", 
        base_url="",
    )
    
    line, id = args
    
    # info = f"scene:{line['data']['scene']}\ndescription:{line['data']['description']}\n"
    
    if "info" not in line.keys() and ("speaker" in line.keys()) and 'emotion' in line['speaker'].keys():
        info = f"emotion: {line['speaker']['emotion']}\nfeeling: {line['speaker']['feeling']}\nneed:{line['speaker']['need']}\n"
    else:
        info = f"scene:{line['speaker']['scene']}\ndescription:{line['speaker']['description']}\n"
    
    dialog = ""
    for i in line["dialogue"]:
        if i['role'] == 'user':
            dialog += f"seeker: {i['content']}\n"
        elif i['role'] == 'assistant':
            dialog += f"supporter: {i['content']}\n"
        else:
            raise Exception("role error")
    
    score = {i:0 for i in range(1, 11)}
    tryCnt = 0
    
    while True and tryCnt <= 4:
        try:
            res = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {'role': 'user', 'content': promptEn.format(diag=dialog, info=info)}],
                n=1,
                temperature=0.0
            )
            cnt = 0
            for i in range(0, 1):
                resText = res.choices[i].message.content
                tmpScore = []
                ck = 1
                for j in [fr'({i})[.:：]+\s*(\d)' for i in range(1,11)]:
                    tmp = re.findall(j, resText)
                    if len(tmp) != 0:
                        tmpScore.append(tmp[0])
                    else:
                        ck = 0
                if ck == 1:
                    cnt += 1
                    for j in tmpScore:
                        score[int(j[0])] += int(j[1])
            for i in score.keys():
                score[i] = score[i]/cnt
            line.update({'comfort':score})
            return line
        
        except Exception as e:
            tryCnt += 1
            print(resText)
            # print(f"{line['info']['id']} - {e}")
            print(f"{id} - {e}")
    line.update({'comfort':{}})
    return line
        
with open(f"/home/hzli/code/ActiveListening/eval/comfort-{dataPath.split('/')[-1].split('.')[0]}-score.json", "w") as file:
    tasks = []
    total = {i:0 for i in range(1, 11)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for id, line in enumerate(data):
            tasks.append(executor.submit(session, (line, id)))
        bar = tqdm(concurrent.futures.as_completed(tasks), total=len(tasks))
        for future in bar:
            res = future.result()
            file.writelines(json.dumps(res, ensure_ascii=False)+"\n")
            file.flush()
            bar.set_description(f"{res['comfort']}")
            
            for i in res['comfort'].keys():
                if i in [7, 10]:
                    total[i] += (7 - res['comfort'][i] + 1)
                else:
                    total[i] += res['comfort'][i]
    
scoreDict = {'情绪改善':[1,2,3,4,5],'帮助动机':[6,7,8,9,10]}
id2dim = {j:i for i in scoreDict.keys() for j in scoreDict[i]}
dimScore = {'情绪改善':0, '帮助动机':0}

for i in total.keys():
    dimScore[id2dim[i]] += total[i]

print(dataPath)
for i in dimScore.keys():
    print(f"{i} - {dimScore[i]/(len(data)*5)}")
    
    
               
                
        