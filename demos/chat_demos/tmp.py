import socket

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


db = SQLAlchemy()


# 创建用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    user_hobby = db.Column(db.String(200))

    def __repr__(self):
        return f'<User {self.username}>'


DATABASE_URI = 'sqlite:////Users/meijiawei/Documents/metabci/emotion_metabci/demos/chat_demos/instance/users.db'


# 创建数据库引擎
engine = create_engine(DATABASE_URI)

# 创建Session类
Session = sessionmaker(bind=engine)

# 创建Session实例
session = Session()

# 查询所有用户
users = session.query(User).all()

# 打印用户信息
for user in users:
    print(f'ID: {user.id}, user_name: {user.username}, hobby: {user.user_hobby}')

