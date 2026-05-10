# Python Flask数据库SQLAlchemy


## 🗄️ Flask 数据库 SQLAlchemy


Flask-SQLAlchemy 配置与模型定义、CRUD 操作、Flask-Migrate 数据库迁移、多表关系（一对多/多对多）、查询过滤与分页。


## Flask-SQLAlchemy 配置


```
// ========== Flask-SQLAlchemy ==========
// 安装: pip install flask-sqlalchemy
// 数据库驱动: pip install psycopg2-binary  # PostgreSQL
//            pip install pymysql          # MySQL
// SQLite 无需额外驱动

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# 配置:
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
# PostgreSQL: "postgresql://user:pass@localhost/dbname"
# MySQL:      "mysql+pymysql://user:pass@localhost/dbname"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ECHO"] = True  # 打印 SQL

db = SQLAlchemy(app)
```


## 模型定义


```
// ========== 模型 ==========
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    age = db.Column(db.Integer, default=18)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # 关系:
    posts = db.relationship("Post", back_populates="author", lazy="dynamic")

    def __repr__(self):
        return f""

class Post(db.Model):
    __tablename__ = "posts"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    author = db.relationship("User", back_populates="posts")

    def __repr__(self):
        return f""

# 创建所有表:
with app.app_context():
    db.create_all()
```


## CRUD 操作


```
// ========== Create ==========
from flask import Flask
from models import db, User, Post

@app.route("/users/create")
def create_user():
    user = User(username="Alice", email="alice@example.com", age=25)
    db.session.add(user)
    db.session.commit()     # 提交事务
    return f"创建用户: {user.id}"

# 批量:
user1 = User(username="Bob", email="bob@example.com")
user2 = User(username="Charlie", email="charlie@example.com")
db.session.add_all([user1, user2])
db.session.commit()

// ========== Read ==========
# 获取所有:
users = User.query.all()
user = User.query.get(1)              # 按主键

# 过滤:
users = User.query.filter_by(username="Alice").all()
users = User.query.filter(User.age > 18).all()
users = User.query.filter(User.email.like("%@example.com")).all()

# 排序:
users = User.query.order_by(User.created_at.desc()).all()

# 分页:
page = User.query.paginate(page=1, per_page=10, error_out=False)
print(page.items)     # 当前页数据
print(page.total)     # 总数
print(page.pages)     # 总页数

# 限制:
users = User.query.limit(5).offset(0).all()

// ========== Update ==========
user = User.query.get(1)
user.age = 26
db.session.commit()

// ========== Delete ==========
user = User.query.get(1)
db.session.delete(user)
db.session.commit()
```


## Flask-Migrate 数据库迁移


```
// ========== Flask-Migrate ==========
// 安装: pip install flask-migrate

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db = SQLAlchemy(app)
migrate = Migrate(app, db)

// 命令行操作:
# 1. 初始化迁移仓库:
flask db init                  # 创建 migrations/ 目录

# 2. 生成迁移脚本 (检测模型变化):
flask db migrate -m "添加用户表"

# 3. 应用迁移:
flask db upgrade               # 更新数据库

# 4. 回滚:
flask db downgrade             # 回滚上一次迁移

# 其他:
flask db history               # 查看迁移历史
flask db current               # 查看当前版本

# 迁移示例:
# 1. 修改模型 (添加字段)
# 2. flask db migrate -m "添加 email 字段"
# 3. flask db upgrade

# 自动生成的迁移文件在 migrations/versions/
```


## 多表关系


```
// ========== 一对多 ==========
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    posts = db.relationship("Post", back_populates="author")

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    author = db.relationship("User", back_populates="posts")

# 使用:
user = User.query.get(1)
for post in user.posts:
    print(post.title)

// ========== 多对多 ==========
# 关联表:
student_course = db.Table("student_course",
    db.Column("student_id", db.Integer, db.ForeignKey("students.id")),
    db.Column("course_id", db.Integer, db.ForeignKey("courses.id"))
)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    courses = db.relationship("Course",
        secondary=student_course,
        back_populates="students"
    )

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    students = db.relationship("Student",
        secondary=student_course,
        back_populates="courses"
    )

// ========== 一对一 ==========
class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bio = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True)
    user = db.relationship("User", back_populates="profile", uselist=False)

# User 中添加: profile = db.relationship("Profile", back_populates="user", uselist=False)
```


> **Note:** 💡 SQLAlchemy 要点: (1) db.Model 定义模型,db.Column 定义字段; (2) CRUD: db.session.add/commit/delete; (3) 查询: filter_by/filter/order_by/limit/paginate; (4) Flask-Migrate 管理数据库版本变更; (5) relationship 定义表关系,ForeignKey 定义外键。


## 练习


<!-- Converted from: 87_Python Flask数据库SQLAlchemy.html -->
