from flask import Flask, render_template, request, session, redirect, url_for
import sqlite3
from chatglm import gradio_interface
import threading
import multiprocessing as mp

def web_server():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key'

    @app.route('/')
    def index():
        if 'username' in session:
            return redirect(url_for('chat'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            # Validate username and password (check against SQLite database)
            # If valid, store username in session
            session['username'] = username
            return redirect(url_for('chat'))
        return render_template('login.html')

    @app.route('/chat')
    def chat():
        if 'username' not in session:
            return redirect(url_for('login'))
        # Render chat interface with chat history
        return render_template('chat.html')

    @app.route('/logout')
    def logout():
        session.pop('username', None)
        return redirect(url_for('login'))

    # Example function for chatbot response generation
    def generate_response(user_input):
        # Implement your chatbot logic here
        # Example:
        if user_input.lower() == 'hello':
            return "Hi there!"
        else:
            return "I didn't understand that."

    # Example function to store chat history in SQLite
    def log_chat(username, message):
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute('INSERT INTO chat_history (username, message) VALUES (?, ?)', (username, message))
        conn.commit()
        conn.close()
    app.run()


# thread = threading.Thread(target=gradio_interface())
# thread.start()
if __name__ == '__main__':
    processes = []
    processes.append(mp.Process(target=gradio_interface))
    processes.append(mp.Process(target=web_server))
    for process in processes:
        process.start()
    # mp.spawn()
    print("here")
    for process in processes:
            process.join()
