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

with open("./prompt/racScale.md", "r") as file:
    promptEn = file.read()

dataPath = ".json"

with open(dataPath, "r") as f:
    data = json.load(f)
        
def session(args):
    # time.sleep(0.5)
    
    client = OpenAI(
        api_key="sk-##", 
        base_url="",
    )
    
    line, id = args
    
    # info = f"scene:{line['speaker']['scene']}\ndescription:{line['speaker']['description']}\n"
    # info = f"emotion:{line['data']['cot_data']['emotion']}\nemotion_stimuli:{line['data']['cot_data']['emotion_stimuli']}\nindividual_appraisal:{line['data']['cot_data']['individual_appraisal']}\n"
    if "info" not in line.keys() and 'emotion' in line['speaker'].keys():
        info = f"emotion: {line['speaker']['emotion']}\nfeeling: {line['speaker']['feeling']}\nneed:{line['speaker']['need']}\n"
    # else:
    #     info = line['info']
    
    dialog = ""
    for i in line['dialogue']:
        dialog += f"{i['role']}: {i['content']}\n"
    
    # for i in line['data']['content']:
    #     if "User" in i.keys():
    #         dialog += f"seeker: {i['User']}\n"
    #     else:
    #         dialog += f"supporter: {i['AI']}\n"
    
    score = {i:0 for i in range(1, 17)}
    tryCnt = 0
    
    while True and tryCnt <= 2:
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
                for j in [fr'({i})[.:：]+\s*(\d)' for i in range(1,17)]:
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
            line.update({'rac':score})
            return line
        
        except Exception as e:
            tryCnt += 1
            print(resText)
            # print(f"{line['info']['id']} - {e}")
            print(f"{id} - {e}")
    line.update({'rac':{}})
    return line
        
with open(f"./eval/rac-{dataPath.split('/')[-1].split('.')[0]}-score.json", "w") as file:
    tasks = []
    total = {i:0 for i in range(1, 17)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for id, line in enumerate(data):
            tasks.append(executor.submit(session, (line, id)))
        bar = tqdm(concurrent.futures.as_completed(tasks), total=len(tasks))
        for future in bar:
            res = future.result()
            file.writelines(json.dumps(res, ensure_ascii=False)+"\n")
            file.flush()
            bar.set_description(f"{res['rac']}")
            
            for i in res['rac'].keys():
                if i in [4]:
                    total[i] += (7 - res['rac'][i] + 1)
                else:
                    total[i] += res['rac'][i]

print(f"./eval/rac-{dataPath.split('/')[-1].split('.')[0]}-score.json")

scoreDict = {'支持性':[i for i in range(1,12)], '对话管理':[12,13,14,15,16]}
id2dim = {j:i for i in scoreDict.keys() for j in scoreDict[i]}
dimScore = {'支持性':0, '对话管理':0}

for i in total.keys():
    dimScore[id2dim[i]] += total[i]
for i in dimScore.keys():
    print(f"{i} - {dimScore[i]/(len(data)*len(scoreDict[i])):.2f}")
        