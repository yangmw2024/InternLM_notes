轻松玩转书生·浦语大模型趣味 Demo
=
1.部署 InternLM2-Chat-1.8B 模型进行智能对话
-
主要部分代码 <br>
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

运行结果： <br>
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_1.png) <br>

第一期用同样internlm2-chat-1_8b模型运行的web_demo结果如下： <br>
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_1_before.png) <br>


2.部署实战营优秀作品 八戒-Chat-1.8B 模型
-
端口配置： <br>
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_2_port.png) <br>

运行结果： <br>
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_2.png) <br>

3.使用 Lagent 运行 InternLM2-Chat-7B 模型
-
Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。<br>
Lagent 的特性总结如下：<br>
(1)流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。<br>
(2)接口统一，设计全面升级，提升拓展性，包括：<br>
1.Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；<br>
2.Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；<br>
3.Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；<br>
(3)文档全面升级，API 文档全覆盖。<br>

examples/internlm2_agent_web_demo_hf.py <br>

结果：
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_3.png) <br>
第一期使用internlm2-chat-1_8b模型结果：
![image](https://github.com/yangmw2024/InternLM_notes/blob/main/IMG/lesson2_3_before.png) <br>


