from zhipuai import ZhipuAI  # 需要将SDK更新到最新版本。
client = ZhipuAI(api_key="bd111f3f2fadb2e914ed1a55ce1453c6.79ZDHSS4xBMi4Sgw") # 填写您自己的APIKey
response = client.chat.completions.create(
    model="emohaa",  # 填写需要调用的模型名称
    meta= {
        "user_info": "30岁的男性软件工程师，兴趣包括阅读、徒步和编程",
        "bot_info": "Emohaa是一款基于Hill助人理论的情感支持AI，拥有专业的心理咨询话术能力",
        "bot_name": "Emohaa",
        "user_name": "张三"
    },
    messages= [
        {
            "role": "assistant",
            "content": "你好，我是Emohaa，很高兴见到你。请问有什么我可以帮忙的吗？"
        },
        {
            "role": "user",
            "content": "最近我感觉压力很大，情绪总是很低落。"
        },
        {
            "role": "assistant",
            "content": "听起来你最近遇到了不少挑战。可以具体说说是什么让你感到压力大吗？"
        },
        {
            "role": "user",
            "content": "主要是工作上的压力，任务太多，总感觉做不完。"
        }
    ]
)
print(response.choices[0].message)