import argparse
import traceback
from utils import *
from typing import *
from tqdm import tqdm
from model import Model, OpenAIModel, ZhiPuModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class Speaker:
    def __init__(self, SpeakerConfig, model: Model, template_path: str):
        self.emotion = SpeakerConfig["emotion"]
        self.need = SpeakerConfig["need"]
        self.memory = SpeakerConfig["memory"]
        self.feeling = SpeakerConfig["feeling"]
        # self.emotion_en = SpeakerConfig["en"].split('\n')[0].split(':')[1].strip()
        self.model = model
        self.template = read_str(template_path)
        # self.system_prompt = self.template.format(en = SpeakerConfig["en"]) # en version
        self.system_prompt = self.template.format(emotion = self.emotion, need = self.need, memory = self.memory, feeling = self.feeling)
        # self.system_prompt = self.template.format(emotion = self.emotion, need = self.need, feeling = self.feeling) # ablation 1
        # self.system_prompt = self.template.format(emotion = self.emotion, feeling = self.feeling) # ablation 2
        self.role_zh = "倾诉者"
        self.role_en = "Speaker"

    def get_response(self, history: Union[str, list]):
        return self.model.get_response(history)


class Listener:

    def __init__(self, model: Model, system_path: str = None):
        self.model = model
        self.system_prompt = None
        self.role_zh = "倾听者"
        self.role_en = "Listener"
        self.advices = []
        self.strategies = []
        if system_path:
            self.system_prompt = read_str(system_path)
        else:
            self.system_prompt = None
    
    def get_response(self, history: Union[str, list]):
        return self.model.get_response(history, self.system_prompt)

class ESCoTListener(Listener):
    def __init__(self, model: Model, system_path = None):
        super().__init__(model, system_path)

    def get_response(self, history):
        s = "Generate the response as the supporter using the pipeline of Emotion, Emotion Stimulus, Individual Appraisal, Strategy Reason, Response.\n\n"
        # print(s)
        if isinstance(history, list):
            for i in history:
                if i['role'] == 'user':
                    s += 'seeker: ' + i['content'] + '\n'
                elif i['role'] == 'assistant':
                    s += 'supporter: ' + i['content'] + '\n'
        response = self.model.get_response(s, self.system_prompt)
        return response.split('\n')[-1]

class SoulChatListener(Listener):
    def __init__(self, model: Model, system_path = None):
        super().__init__(model, system_path)
    
    def get_response(self, history):
        s = "现在请你扮演一位心理咨询师，与用户展开对话。\n\n"
        # print(s)
        if isinstance(history, list):
            for i in history:
                if i['role'] == 'user':
                    s += '用户：' + i['content'] + '\n心理咨询师：'
                elif i['role'] == 'assistant':
                    s += i['content'] + '\n'
        response = self.model.get_response(s, self.system_prompt)
        return response

class MeChatListener(Listener):
    def __init__(self, model: Model, system_path = None):
        super().__init__(model, system_path)
    
    def get_response(self, history):
        instruction = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
'''
        # print(s)
        if isinstance(history, list):
            for i in history:
                if i['role'] == 'user':
                    instruction += '来访者：' + i['content'] + '\n咨询师：'
                elif i['role'] == 'assistant':
                    instruction += i['content'] + '\n'
        response = self.model.get_base_response(instruction, self.system_prompt)
        return response

class LlamaListener(Listener):
    def __init__(self, model: Model, system_path: str = None):
        super().__init__(model, system_path)
    
    def get_response(self, history: Union[str, list]):
        return self.model.get_response(([{'role':'system', 'content':'你是一名精通中文的助手。'}, {"role": "user", "content": self.system_prompt + history}]))

class ChatHistory:
    def __init__(self):
        self.str = ""
        self.ListenerHistory = []
        self.SpeakerHistory = []

    def add_str(self, role, content):
        if content.startswith(f"{role}: "):
            self.str += content + '\n'
        else:
            self.str += f'{role}: ' + content + '\n'

class ModelChat:
    def __init__(
            self,
            speaker_model,
            listener_model: Listener,
            maxRound: int,
            log: bool = False
        ):
        self.Round = 0
        self.maxRound = maxRound
        self.speaker = speaker_model
        self.listener = listener_model
        self.log = log
        self.history = ChatHistory()
        self.init_history()
        
    def init_history(self):
        prompt0 = f'你看上去好像有点难过？'
        self.history.add_str(self.listener.role_zh, prompt0)
        self.history.SpeakerHistory += [{'role': 'system', 'content': self.speaker.system_prompt}] + [{'role': 'user', 'content': prompt0}]

        if self.listener.system_prompt is not None:
            self.history.ListenerHistory += [{'role': 'system', 'content': self.listener.system_prompt}] + [{'role': 'assistant', 'content': prompt0}]
        else:
            self.history.ListenerHistory += [{'role': 'assistant', 'content': prompt0}]

    def step(self):
        try: 
            done = 0
            
            # Speaker History
            speaker_response = self.speaker.get_response(history = self.history.SpeakerHistory)
            self.history.add_str(self.speaker.role_zh, speaker_response)
            self.history.SpeakerHistory += [{'role': 'assistant', 'content': speaker_response}]
            self.history.ListenerHistory += [{'role': 'user', 'content': speaker_response}]
            
            # Listener History
            listener_response = self.listener.get_response(history = self.history.ListenerHistory)
            self.history.add_str(self.listener.role_zh, listener_response)
            self.history.SpeakerHistory += [{'role': 'user', 'content': listener_response}]
            self.history.ListenerHistory += [{'role': 'assistant', 'content': listener_response}]

            self.Round += 1
            if self.log:
                print(self.history.ListenerHistory)

            return done, self.history, self.listener.advices, self.listener.strategies
        except Exception as e:
            print(e)
            traceback.print_exc()
            done = 1
            return done, self.history, self.listener.advices, self.listener.strategies

    def chatchat(self):
        for i in range(self.maxRound):
            done, history, advices, strategies = self.step()
            if done:
                break
        return history, advices, strategies

def llamaBaseline(args):
    Speaker_config, t = args
    config = read_yaml("yaml/APIConfig.yaml")
    Openai = config["openai"]
    llama = config["LLaMA"]

    speakerModel = OpenAIModel("yaml/SpeakerModel.yaml", Openai["api_key"], Openai["base_url"])
    speaker = Speaker(Speaker_config, speakerModel, "prompt/speaker_active.md")

    listenerModel = OpenAIModel("yaml/llamaModel.yaml", llama["api_key"], llama["base_url"])
    listener = Listener(listenerModel, "prompt/listener_model.md")

    maxTurn = config["maxTurn"]
    chat = ModelChat(speaker, listener, maxTurn, log = True)
    history, advices, strategies = chat.chatchat()

    return Speaker_config, history, advices, strategies

def ESCoTBaseline(args):
    Speaker_config, t = args
    config = read_yaml("yaml/APIConfig.yaml")
    Openai = config["openai"]
    ESCoT = config["ESCoT"]

    speakerModel = OpenAIModel("yaml/SpeakerModel.yaml", Openai["api_key"], Openai["base_url"])
    speaker = Speaker(Speaker_config, speakerModel, "prompt/speaker_active_en.md")

    listenerModel = OpenAIModel("yaml/ESCoT.yaml", ESCoT["api_key"], ESCoT["base_url"])
    listener = ESCoTListener(listenerModel)

    maxTurn = config["maxTurn"]
    chat = ModelChat(speaker, listener, maxTurn)

    history, advices, strategies = chat.chatchat()

    return Speaker_config, history, advices, strategies

def SoulChatBaseline(args):
    Speaker_config, t = args
    config = read_yaml("yaml/APIConfig.yaml")
    Openai = config["openai"]
    SoulChat = config["SoulChat"]

    speakerModel = OpenAIModel("yaml/SpeakerModel.yaml", Openai["api_key"], Openai["base_url"])
    speaker = Speaker(Speaker_config, speakerModel, "prompt/speaker_active.md")

    listenerModel = OpenAIModel("yaml/SoulChat.yaml", SoulChat["api_key"], SoulChat["base_url"])
    listener = SoulChatListener(listenerModel)

    maxTurn = config["maxTurn"]
    chat = ModelChat(speaker, listener, maxTurn)

    history, advices, strategies = chat.chatchat()

    return Speaker_config, history, advices, strategies

def MeChatBaseline(args):
    Speaker_config, t = args
    config = read_yaml("yaml/APIConfig.yaml")
    Openai = config["openai"]
    MeChat = config["MeChat"]

    speakerModel = OpenAIModel("yaml/SpeakerModel.yaml", Openai["api_key"], Openai["base_url"])
    speaker = Speaker(Speaker_config, speakerModel, "prompt/speaker_active.md")

    listenerModel = OpenAIModel("yaml/MeChat.yaml", MeChat["api_key"], MeChat["base_url"])
    listener = MeChatListener(listenerModel)

    maxTurn = config["maxTurn"]
    chat = ModelChat(speaker, listener, maxTurn)

    history, advices, strategies = chat.chatchat()

    return Speaker_config, history, advices, strategies

def EmoLLMBaseline(args):
    Speaker_config, t = args
    config = read_yaml("yaml/APIConfig.yaml")
    Openai = config["openai"]
    EmoLLM = config["EmoLLM"]

    speakerModel = OpenAIModel("yaml/SpeakerModel.yaml", Openai["api_key"], Openai["base_url"])
    speaker = Speaker(Speaker_config, speakerModel, "prompt/speaker_active.md")

    listenerModel = OpenAIModel("yaml/EmoLLM.yaml", EmoLLM["api_key"], EmoLLM["base_url"])
    listener = Listener(listenerModel)

    maxTurn = config["maxTurn"]
    chat = ModelChat(speaker, listener, maxTurn)

    history, advices, strategies = chat.chatchat()

    return Speaker_config, history, advices, strategies

def testConnection():
    APICONFIG = read_yaml("yaml/APIConfig.yaml")
    MeChat = APICONFIG["MeChat"]
    listenerModel = OpenAIModel("yaml/MeChat.yaml", MeChat["api_key"], MeChat["base_url"])
    listener = MeChatListener(listenerModel)

    print(listener.get_response([{'role': 'user', 'content': "你好"}]))
    exit(0)

if __name__ == "__main__":

    # testConnection()

    dia = []
    t_str = time_format()

    parser = argparse.ArgumentParser(description="传入baseline类型，仅空/cot/planner/draft可选")
    parser.add_argument('--type', type=str, required=True, help="baseline方法类型")
    parser.add_argument('--model', type=str, required=True, help="哪个模型的baseline")
    args = parser.parse_args()
    assert args.type in ["", "cot", "planner", "draft"]

    cot = args.type # "" or "cot"
    if args.model == "llama":
        Baseline = llamaBaseline
    elif args.model == "ESCoT":
        Baseline = ESCoTBaseline
    elif args.model == "SoulChat":
        Baseline = SoulChatBaseline
    elif args.model == "EmoLLM":
        Baseline = EmoLLMBaseline
    elif args.model == "MeChat":
        Baseline = MeChatBaseline
    else:
        raise ValueError("model not found")
    listener_model = Baseline.__name__ + cot
    
    speakers = read_json("./data/.json")
    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = []
        for speaker in speakers:
            config = speaker["speaker"]
            tasks.append(executor.submit(Baseline, (config, args.type)))
        for future in tqdm(as_completed(tasks), total=len(tasks)):    
            dia.append({
                "speaker": future.result()[0],
                "dialog": future.result()[1].ListenerHistory,
                "advices": future.result()[2],
                "strategies": future.result()[3],
                "dialog_str": future.result()[1].str
            })
            write_json(f"./eval/testDialog/{listener_model.split('/')[-1]}_result_{t_str}.json", dia)

    print(f"saved in ./eval/testDialog/{listener_model.split('/')[-1]}_result_{t_str}.json")

