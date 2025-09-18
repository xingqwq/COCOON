import numpy as np
import json
import re
import random
import pandas as pd
import yaml

def time_format(format="%m-%d-%H-%M"):
    from datetime import datetime
    now = datetime.now()
    return now.strftime(format)

def write_json(path, s):
    str_s = json.dumps(s, ensure_ascii = False, indent = 4)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str_s)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        str_s = json.load(f)
    return str_s

def read_jsonline(path):
    Ldic = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            Ldic.append(json.loads(line))
    return Ldic

def write_jsonline(json_name, s):
    with open('./' + json_name, 'w', encoding='utf-8') as f:   
        for page in s:
            json.dump(page, f, ensure_ascii=False)
            f.write('\n')

def read_npy(path):
    return np.load(path, allow_pickle = True)

def read_csv(path):
    df = pd.read_csv(path, encoding='gbk')
    return df.to_dict(orient='records')

def read_txt(path): #split by \n
    out = []
    with open(path, 'r') as f:
        for line in f:
            out.append(line)
    return out

def read_str(path):
    with open(path, 'r') as f:
        s = f.read()
    return s

def read_yaml(path):
    with open(path, "r") as f:
        s = yaml.safe_load(f)
    return s