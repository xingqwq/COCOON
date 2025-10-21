from utils import *
import time
import torch
from openai import OpenAI
import openai
from zhipuai import ZhipuAI
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig
from typing import *

class Config:
    def __init__(self, config: Union[dict, str]):
        if isinstance(config, str):
            self.config = read_yaml(config)
        else:
            self.config = config
        self.model_name = self.config["model"]
        self.generationConfig = self.config.get("generationConfig", {})

class Model:
    def __init__(self, generation_config: Config):
        if isinstance(generation_config, Config):
            self.generation_config = generation_config.config
        elif isinstance(generation_config, (str, dict)):
            self.generation_config = Config(generation_config)
        else:
            raise ValueError("generation_config must be a dict or a yaml file path or a Config object.")

    def get_response(self) -> str:
        pass


class MeChat(Model):
    def __init__(self, generation_config: Config, model_path):
        super().__init__(generation_config)
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().cuda()
        model = model.eval()
        return model, tokenizer
    
    def get_dialogue_history_en(self, history):
        dialogue_history_tmp = []
        for item in history:
            if item['role'] == 'assistant':
                text = 'counselor: '+ item['content']
            elif item['role'] == 'user':
                text = 'client: '+ item['content']
            else:
                continue
            dialogue_history_tmp.append(text)

        dialogue_history = '\n'.join(dialogue_history_tmp)

        return dialogue_history + '\n' + 'counselor: '

    def get_instruction_en(self, dialogue_history):
        instruction = f'''Now you will act as a professional psychological counselor with extensive knowledge in psychology and mental health. You excel in employing various counseling techniques, such as principles of Cognitive Behavioral Therapy (CBT), Motivational Interviewing, and solution-focused brief therapy. Using a warm and empathetic tone, you demonstrate deep understanding and compassion for the client's feelings.

Engage in a natural conversational manner, avoiding responses that are too short or overly detailed, ensuring a smooth and human-like flow. Provide in-depth guidance and insights, using specific psychological concepts and examples to help clients explore their thoughts and emotions more deeply. Avoid sounding instructive, focusing instead on empathy and respecting the client’s emotions. Adapt your responses based on the client’s feedback to ensure they are relevant to their situation and needs. Please generate a reply for the following conversation.

Conversation：
{dialogue_history}'''

        return instruction

    def get_dialogue_history(dialogue_history_list: list):
        dialogue_history_tmp = []
        for item in dialogue_history_list:
            if item['role'] == 'counselor' or item['role'] == 'assistant':
                text = '咨询师：'+ item['content']
            else:
                text = '来访者：'+ item['content']
            dialogue_history_tmp.append(text)
        dialogue_history = '\n'.join(dialogue_history_tmp)
        return dialogue_history + '\n' + '咨询师：'

    def get_instruction(self, dialogue_history):
        instruction = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
{dialogue_history}'''

        return instruction

    def get_response(self, history, system_prompt: str = None):
        if system_prompt and history[0]['role'] != 'system':
            history = [{'role': 'system', 'content': system_prompt}] + history
        dialogue_history = self.get_dialogue_history(history)
        instruction = self.get_instruction(dialogue_history)
        response, history = self.model.chat(self.tokenizer, instruction, history=[], **self.generation_config.generationConfig)
        return response

class QwenModel(Model):
    def __init__(self, generation_config: Config, model_path):
        super().__init__(generation_config)
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model()
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True
        )
        return model, tokenizer

    def get_response(self, history, system_prompt: str = None):
        if system_prompt and history[0]['role'] != 'system':
            history = [{'role': 'system', 'content': system_prompt}] + history
        text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            **self.generation_config.generationConfig
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class OpenAIModel(Model):
    def __init__(self, generation_config: Config, api_key, base_url, tokenizer = None):
        super().__init__(generation_config)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key = self.api_key, base_url = self.base_url)
        self.tokenizer = tokenizer
        if self.tokenizer is not None and isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def get_response(self, history: Union[list, str], system_prompt: str = None) -> str:
        """
        Get a response (chat or base) based on whether a tokenizer is provided or not.

        Args:
            history (list or str): A list of conversation messages or a string.

        Returns:
            str: The generated response.
        """
        if self.tokenizer is None:
            return self.get_chat_response(history, system_prompt)
        else:
            return self.get_base_response(history, system_prompt)

    def get_chat_response(self, history: Union[list, str], system_prompt: str = None):
        messages = history
        if isinstance(history, str):
            messages = [{'role': 'user', 'content': history}]
        if system_prompt and messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': system_prompt}] + messages
        max_retries = 3
        model = self.generation_config.model_name
        for attempt in range(max_retries):   
            try: 
                response = self.client.chat.completions.create(
                    model = model,
                    messages = messages,
                    **self.generation_config.generationConfig
                )
                # print(response.choices[0].message)
                response = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                # print(response)
                # print(messages)
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(3)
        return response

    def get_base_response(self, history: Union[list, str], system_prompt: str = None):
        messages = history
        if isinstance(history, str):
            messages = [{'role': 'user', 'content': history}]
        if system_prompt and messages[0]['role'] != 'system':
            messages = [{'role': 'system', 'content': system_prompt}] + messages
        max_retries = 3
        model = self.generation_config.model_name
        if self.tokenizer is not None:
            history_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            history_str = history if isinstance(history, str) else '\n'.join([f'{m["content"]}' for m in history])
        for attempt in range(max_retries):   
            try: 
                response = self.client.completions.create(
                    model = model,
                    prompt = history_str,
                    **self.generation_config.generationConfig
                )
                response = response.choices[0].text.strip()
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(1)
        return response
    
class ZhiPuModel(Model):
    def __init__(self, generation_config: Config, config_path = 'API_config.yaml'):
        super().__init__(generation_config)
        self.api_key = read_yaml(config_path)['ZhipuAI']["api_key"]
        self.client = ZhipuAI(api_key = self.api_key)

    def get_response(self, history = [], system_prompt = None):
        if system_prompt and history[0]['role'] != 'system':
            history = [{'role': 'system', 'content': system_prompt}] + history
        response = self.client.chat.completions.create(
            model = "emohaa",
            meta = {
                "bot_info": "Emohaa是一款基于Hill助人理论的情感支持AI，拥有专业的心理咨询话术能力",
                "bot_name": "Emohaa",
            },
            messages = history,
            **self.generation_config.generationConfig
        )
        return response.choices[0].message.content.strip()