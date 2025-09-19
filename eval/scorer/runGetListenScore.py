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
    
with open("./prompt/listen.md", "r") as file:
    promptEn = file.read()

with open("./prompt/listen.md", "r") as file:
    promptZh = file.read()

dataPath = ""
with open(dataPath, "r") as file:
    data = file.readlines()
        
        
def session(args):
    # time.sleep(0.5)
    
    client = OpenAI(
        api_key="sk-##", 
        base_url="",
    )
    
    line, id = args
    if "info" not in line.keys():
        info = f"emotion: {line['emotion']}\nfeeling: {line['feeling']}\nneed:{line['need']}\n"
    else:
        info = line['info']
        
    score = {i:0 for i in range(1, 11)}
    tryCnt = 0
    
    while True and tryCnt <= 4:
        try:
            res = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {'role': 'user', 'content': promptZh.format(diag=line['dialog'], info=info)}],
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
            return {'lineId':id, 'dialog':line['dialog'], 'info':info, 'score':deepcopy(score)}
        
        except Exception as e:
            tryCnt += 1
            print(resText)
            # print(f"{line['info']['id']} - {e}")
            print(f"{id} - {e}")
    
    return {'lineId':id, 'dialog':line['dialog'], 'info':info, 'score':deepcopy(score)}
        
with open(f"listen-{dataPath.split('/')[-1].split('.')[0].split('-')[0]}-score.json", "w") as file:
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for id, line in enumerate(data):
            line = json.loads(line)
            tasks.append(executor.submit(session, (line, id)))
        bar = tqdm(concurrent.futures.as_completed(tasks), total=len(tasks))
        for future in bar:
            res = future.result()
            file.writelines(json.dumps(res, ensure_ascii=False)+"\n")
            file.flush()
            bar.set_description(f"{res['score']}")
        
