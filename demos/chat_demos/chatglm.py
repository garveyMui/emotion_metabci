from zhipuai import ZhipuAI  # 需要将SDK更新到最新版本。
import gradio as gr
from langchain.schema import AIMessage, HumanMessage

client = ZhipuAI(api_key="bd111f3f2fadb2e914ed1a55ce1453c6.79ZDHSS4xBMi4Sgw")  # 填写您自己的APIKey

def predict(message, history, assistant_prompt="", is_stream=False):
    messages_input = []
    for human, assistant in history[:-1]:
        messages_input.append({"role": "user", "content": human})
        messages_input.append({"role": "assistant", "content": assistant})
    messages_input.append({"role": "user", "content": history[-1][0]})

    generate_kwargs = {
        "max_tokens": 1024,
        "do_sample": True,
        # "top_p": 0.8,
        "temperature": 0.6,
    }
    demo_message = [
            {
                "role": "assistant",
                "content": "你好，我是emohaa，很高兴见到你。请问有什么我可以帮忙的吗？"
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
    response = client.chat.completions.create(
        model="emohaa",  # 填写需要调用的模型名称
        meta={
            "user_info": "30岁的男性软件工程师，兴趣包括阅读、徒步和编程",
            "bot_info": "Emohaa是一款基于Hill助人理论的情感支持AI，拥有专业的心理咨询话术能力",
            "bot_name": "Emohaa",
            "user_name": "张三"
        },
        messages=messages_input,
        stream=is_stream
    )
    if is_stream:
        response_message = ""
        for chunk in response:
            print(chunk.choices[0].delta)
            response_message += chunk.choices[0].delta
    else:
        print(response.choices[0].message)
        response_message = response.choices[0].message.content.strip()
    print(type(response_message))
    # history += [[message, ""]]
    history[-1][1] += response_message
    return history
def gradio_interface():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">GLM-4-9B Gradio Simple Chat Demo</h1>""")
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit")
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(show_label=False, placeholder="Prompt", lines=10, container=False)
                pBtn = gr.Button("Set Prompt")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


        def user(query, history):
            return "", history + [[query, ""]]


        def set_prompt(prompt_text):
            return [[prompt_text, "成功设置prompt"]]


        pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)

        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot]).then(predict, [user_input, chatbot], chatbot, queue=False)
        emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=8000, inbrowser=True, share=True)

