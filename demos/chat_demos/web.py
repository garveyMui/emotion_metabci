from flask import Flask, render_template, request, session, redirect, url_for
import sqlite3
from chatglm import gradio_interface
import threading
import multiprocessing as mp
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import multiprocessing as mp
from collector import collecting_all_info


def web_server():

    app = Flask(__name__)
    app.secret_key = 'your_secret_key'

    app.config['SECRET_KEY'] = 'your_secret_key'

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    db = SQLAlchemy(app)

    # 创建用户模型
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        password = db.Column(db.String(120), nullable=False)
        user_hobby = db.Column(db.String(200))
        def __repr__(self):
            return f'<User {self.username}>'
    class Current(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(80), unique=False, nullable=False)
        info = db.Column(db.String(200), unique=False, nullable=False)

    # 创建数据库表
    with app.app_context():
        db.create_all()

    # 注册页面
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            # password1 = request.form['passwordAgain']
            # if password != password1:
            #     flash('Passwords do not match', 'error')
            #     return render_template('register.html')
            info = request.form['info']
            hashed_password = generate_password_hash(password, method="scrypt")
            try:
                new_user = User(username=username, password=hashed_password, user_hobby=info)
                db.session.add(new_user)
                res = Current.query.filter_by(name="dummy").first()
                if res is None:
                    dummy_current = Current(name="dummy", info="dummy info")
                    db.session.add(dummy_current)
                db.session.commit()
                # db.session.close()
            except Exception as e:
                print("create user failed!!!", e)
                flash('username occupied!', 'error')
                return render_template('register.html')


            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        return render_template('register.html')

    # 登录页面
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            user = User.query.filter_by(username=username).first()
            try:
                user_hobby = user.user_hobby
            except Exception as e:
                print("no hobby found!!!", e)
            else:
                user_hobby = ""
            if user and check_password_hash(user.password, password):
                flash('Login successful.', 'success')
                session['username'] = username

                first_row = Current.query.first()
                first_row.name = user.username
                first_row.info = user.user_hobby
                db.session.commit()

                process = mp.Process(target=collecting_all_info, args=(username, user_hobby))
                process.start()

                return redirect(url_for('chat'))
            else:
                flash('Invalid username or password.', 'danger')
        return render_template('login.html')

    # for test
    @app.route('/dashboard')
    def dashboard():
        return 'Welcome to the dashboard!'

    @app.route('/')
    def index():
        if 'username' in session:
            return redirect(url_for('chat'))
        return redirect(url_for('login'))

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
    # sqlite_db = sqlite3.connect('users.db')
    web_server()