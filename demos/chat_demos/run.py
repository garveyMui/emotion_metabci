import multiprocessing as mp

from demos.chat_demos.chatglm import gradio_interface
from demos.chat_demos.web import web_server

if __name__ == '__main__':
    processes = []
    processes.append(mp.Process(target=gradio_interface))
    processes.append(mp.Process(target=web_server))
    for process in processes:
        process.start()
    # mp.spawn()
    for process in processes:
        process.join()
