import json
from tqdm import tqdm
import time
import os
import random
from openai import OpenAI
from copy import deepcopy
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def write_json(path, s):
    str_s = json.dumps(s, ensure_ascii = False, indent = 4)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str_s)

random.seed(6600)
def all_score(diag):
    return f'''Please evaluate the quality of the conversation between the user and the emotional assistant based on the given criteria. Your task is to assess the conversation using the following seven key indicators and provide a score (1-10).

## Criteria

1. Fluency: Assess the logical flow and structure of the conversation. A score of 1 indicates the conversation is completely incoherent, with a disorganized and unclear expression. A score of 10 indicates the conversation is logically clear and flows smoothly.
2. Diversity: Evaluate the diversity of expressions and the richness of the content in the conversation. A score of 1 means the conversation is stiff, having duplicate content, lacking the ability to absorb and internalize content. A score of 10 indicates the conversation has highly diverse expressions, rich content, and strong expressiveness.
3. Empathy: Assess the assistant's understanding of the user's emotions and whether it helps the user analyze the underlying logic of their emotions, providing emotional support. A score of 1 means no empathy is shown. A score of 10 indicates a high level of empathy, effectively supporting the user.
4. Information: Focus on whether the assistant’s advice is reasonable and sufficient in quantity, ensuring that the information is both helpful and solves the user’s problem. A score of 1 means the assistant’s advice is entirely ineffective and may even harm the user. A score of 10 means the assistant’s advice is effective, abundant (more than 5 suggestions), and helps the user solve their problems very well.
5. Humanoid: Evaluate whether the assistant demonstrates human-like interaction abilities, avoiding stiff, robotic responses or obvious AI language model traces. A score of 1 means the conversation is entirely stiff and inhuman, failing to internalize content. A score of 10 means there are no AI traces, and the conversation feels indistinguishable from talking to a human.
6. Skillful: Assess whether the assistant’s responses exhibit key abilities, including empathy, information quality, hopefulness, importance, and providing necessary suggestions or highlights. A score of 1 means only one or fewer of these abilities are demonstrated, while a score of 10 means all five abilities are present and excellently displayed.
7. Overall: Based on your subjective impression, evaluate the assistant’s overall performance, including whether the user would enjoy and recommend this assistant. A score of 1 means the user does not like the assistant. A score of 10 means the user would highly enjoy chatting with this assistant and would recommend it to friends.

## Format

Only provide the scores, without any explanations, and the scores should be integers between 1 and 10. The format should be as follows:

Fluency: [score]
Diversity: [score]
Empathy: [score]
Information: [score]
Humanoid: [score]
Skillful: [score]
Overall: [score]

[Conversation Record]
{diag}
'''

def sessionAll(args):
    client = OpenAI(
        api_key="sk-##", 
        base_url="",
    )
    
    line, id, diag = args
    prompt = all_score(diag)

    line['EscRank'] = {}
    
    err = 1

    tryCnt = 0
        
    while True and tryCnt <= 2:
        try:
            res = client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {'role': 'user', 'content': prompt}],
                n=1,
                temperature=0.0
            )
            resText = res.choices[0].message.content
            pattern = r"Fluency:\s*(\d{1,2})\s*Diversity:\s*(\d{1,2})\s*Empathy:\s*(\d{1,2})\s*Information:\s*(\d{1,2})\s*Humanoid:\s*(\d{1,2})\s*Skillful:\s*(\d{1,2})\s*Overall:\s*(\d{1,2})"
            matches = re.search(pattern, resText)

            # print(resText)
            # with open("./log.txt", "w") as f:
            #     f.write(resText)

            # x = input()

            if matches:
                scores = {
                    "Fluency": matches.group(1),
                    "Diversity": matches.group(2),
                    "Empathy": matches.group(3),
                    "Information": matches.group(4),
                    "Humanoid": matches.group(5),
                    "Skillful": matches.group(6),
                    "Overall": matches.group(7),
                }
                # print(scores)
                line['EscRank'] = scores
                err = 0
                break
            print(resText)
            with open('./log.txt', 'w') as f:
                f.write(resText)
            x = input()
        
        except Exception as e:
            tryCnt += 1
            print(f"{id} - {e}")
        if err == 1:
            line['EscRank'] = {}
    return line

def construct_ActiveListening(data, user = '倾诉者', assistant = '倾听者'):
    diag = data['dialog']
    d = ''
    for i in diag:
        if i.get('role', "") == user:
            d += f"User: {i['content'].strip()}\n"
        elif i.get('role', "") == assistant:
            d += f"Assistant: {i['content'].strip()}\n"
        elif i.get('role', "") == 'user':
            d += f"User: {i['content'].strip()}\n"
        elif i.get('role', "") == 'assistant':
            d += f"Assistant: {i['content'].strip()}\n"
    return d

def construct_ESC(data, user = 'seeker', assistant = 'supporter'):
    diag = data['dialog']
    d = ''
    for i in diag:
        if i.get('speaker', "") == user:
            d += f"User: {i['content'].strip()}\n"
        elif i.get('speaker', "") == assistant:
            d += f"Assistant: {i['content'].strip()}\n"
    return d

def construct_ESCoT(data):
    diag = data['original_data']["dialog"] + [{'speaker': 'supporter', 'content': data['original_data']['response']}]
    d = ''
    for i in diag:
        if i.get('speaker', "") == 'seeker':
            d += f"User: {i['content'].strip()}\n"
        elif i.get('speaker', "") == 'supporter':
            d += f"Assistant: {i['content'].strip()}\n"
    return d

def construct_ExTES(data):
    diag = data["content"]
    d = ''
    for i in diag:
        for x in i.keys():
            if x == 'User':
                d += f"User: {i[x].strip()}\n"
            elif x == 'AI':
                d += f"Assistant: {i[x].strip()}\n"
    return d

def construct_SoulChat(data):
    diag = data["messages"]
    d = ''
    for i in diag:
        if i.get('role', "") == 'user':
            d += f"User: {i['content'].strip()}\n"
        elif i.get('role', "") == 'assistant':
            d += f"Assistant: {i['content'].strip()}\n"
    return d

def construct_EmoLLM(data):
    dia = data["conversation"]
    d = ''
    for i in dia:
        d += f"User: {i['input']}\n"
        d += f"Assistant: {i['output']}\n"
    return d

def construct_Smile(data, user = 'client', assistant = 'counselor'):
    diag = data['dialog']
    d = ''
    for i in diag:
        if i.get('role', "") == user:
            d += f"User: {i['content'].strip()}\n"
        elif i.get('role', "") == assistant:
            d += f"Assistant: {i['content'].strip()}\n"
        elif i.get('role', "") == 'user':
            d += f"User: {i['content'].strip()}\n"
        elif i.get('role', "") == 'assistant':
            d += f"Assistant: {i['content'].strip()}\n"
    return d

def time_format():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%m-%d_%H-%M")

def main():
    filepath = ""

    with open(filepath, 'r') as f:
        data = json.load(f)

    construct = construct_ActiveListening

    sample = data
    dia = []

    t_str = time_format()

    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = []
        for id, d in enumerate(sample):
            diag = construct(d)
            tasks.append(executor.submit(sessionAll, (d, id, diag)))
        for future in tqdm(as_completed(tasks), total=len(tasks)):    
            dia.append(future.result())
            write_json(f"./ESC-Eval-main/gptscore{t_str}.json", dia)
    calc(dia)

def calc(data):
    score = {}
    scales = ["Fluency", "Diversity", "Empathy", "Information", "Humanoid", "Skillful", "Overall"]
    for s in scales:
        score[s] = 0
        for i in data:
            score[s] += int(i['EscRank'][s])
    print(score)


if __name__ == '__main__':
    main()