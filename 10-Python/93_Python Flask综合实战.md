# Python Flask综合实战


## 🏗️ Flask 综合实战


从零搭建完整 Flask 博客项目：项目结构、用户认证、文章 CRUD、API 构建、测试、Docker 部署。Flask 知识全景回顾。


## 项目结构


```
// ========== 博客项目结构 ==========
# flask-blog/
# ├── app/
# │   ├── __init__.py         # 应用工厂
# │   ├── config.py           # 配置
# │   ├── models/
# │   │   ├── __init__.py
# │   │   ├── user.py         # 用户模型
# │   │   └── post.py         # 文章模型
# │   ├── routes/
# │   │   ├── __init__.py
# │   │   ├── auth.py         # 认证路由
# │   │   ├── posts.py        # 文章路由
# │   │   └── api.py          # API 路由
# │   ├── templates/
# │   │   ├── base.html
# │   │   ├── index.html
# │   │   ├── auth/
# │   │   │   ├── login.html
# │   │   │   └── register.html
# │   │   └── posts/
# │   │       ├── list.html
# │   │       └── detail.html
# │   ├── static/
# │   │   ├── css/style.css
# │   │   └── js/app.js
# │   └── utils/
# │       ├── __init__.py
# │       └── helpers.py      # 辅助函数
# ├── tests/
# │   ├── __init__.py
# │   ├── conftest.py
# │   ├── test_auth.py
# │   └── test_posts.py
# ├── .env
# ├── .env.example
# ├── .gitignore
# ├── Dockerfile
# ├── docker-compose.yml
# ├── requirements.txt
# └── run.py
```


## 应用工厂与模型


```
// ========== app/__init__.py ==========
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def create_app(config_name="app.config.DevelopmentConfig"):
    app = Flask(__name__)
    app.config.from_object(config_name)

    # 初始化扩展:
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # 注册蓝图:
    from app.routes.auth import auth_bp
    from app.routes.posts import posts_bp
    from app.routes.api import api_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(posts_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # 注册错误处理:
    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not Found"}, 404

    return app

# ========== models/user.py ==========
from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    posts = db.relationship("Post", backref="author", lazy="dynamic")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
```


## 路由与视图


```
// ========== routes/auth.py ==========
from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User
from app import db

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user = User(
            username=request.form["username"],
            email=request.form["email"]
        )
        user.set_password(request.form["password"])
        db.session.add(user)
        db.session.commit()
        flash("注册成功!", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.check_password(request.form["password"]):
            login_user(user)
            return redirect(url_for("posts.index"))
        flash("用户名或密码错误", "error")
    return render_template("auth/login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))

# ========== routes/posts.py ==========
posts_bp = Blueprint("posts", __name__)

@posts_bp.route("/")
@posts_bp.route("/page/")
def index(num=1):
    page = Post.query.order_by(Post.created_at.desc()).paginate(
        page=num, per_page=10, error_out=False
    )
    return render_template("posts/list.html", posts=page)

@posts_bp.route("/post/")
def detail(id):
    post = Post.query.get_or_404(id)
    return render_template("posts/detail.html", post=post)
```


## API 与测试


```
// ========== routes/api.py ==========
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.post import Post

api_bp = Blueprint("api", __name__)

@api_bp.route("/posts")
def list_posts():
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return jsonify([{
        "id": p.id, "title": p.title,
        "author": p.author.username,
        "created_at": p.created_at.isoformat()
    } for p in posts])

@api_bp.route("/posts/")
def get_post(id):
    post = Post.query.get_or_404(id)
    return jsonify({
        "id": post.id, "title": post.title,
        "content": post.content, "author": post.author.username
    })

# ========== 测试 conftest.py ==========
import pytest
from app import create_app, db

@pytest.fixture
def app():
    app = create_create_app("app.config.TestConfig")
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

# ========== test_auth.py ==========
def test_register(client):
    resp = client.post("/register", data={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass"
    })
    assert resp.status_code == 302  # 注册后重定向到登录

def test_login(client):
    # 先注册
    client.post("/register", data={
        "username": "testuser", "email": "test@test.com",
        "password": "testpass"
    })
    # 登录
    resp = client.post("/login", data={
        "username": "testuser", "password": "testpass"
    })
    assert resp.status_code == 302
```


## Flask 知识全景


```
// ========== Flask 知识体系 ==========
// 基础:
// 571 - Flask 基础 (路由/request/response/jsonify)
// 572 - 路由与视图 (url_for/Blueprint/工厂模式/钩子)
// 573 - 模板与表单 (Jinja2/WTForms/flash)
// 574 - 数据库 (SQLAlchemy/Migrate/CRUD/关系)
// 575 - REST API (Blueprint/marshmallow/错误处理)

// 进阶:
// 576 - 用户认证 (密码哈希/Flask-Login/JWT/session)
// 577 - 文件上传 (验证/Pillow/扩展生态)
// 578 - 测试 (pytest/fixture/覆盖率)
// 579 - 生产部署 (Gunicorn/Nginx/Docker)
// 580 - 综合实战 (项目结构/完整示例)

// ========== 学习路径 ==========
// 1. 掌握基本路由和模板 (571-573)
// 2. 集成数据库 (574)
// 3. 构建 REST API (575)
// 4. 添加用户认证 (576)
// 5. 深入了解扩展 (577)
// 6. 编写测试 (578)
// 7. 部署上线 (579)
// 8. 综合实战 (580)
```


> **Note:** 💡 Flask 综合要点: (1) 模块化: 应用工厂 + Blueprint + 分层模型; (2) CRUD + Auth + API 覆盖全功能; (3) pytest + conftest.fixture 测试覆盖; (4) Docker + Gunicorn + Nginx 生产部署; (5) Flask 适合中小项目,快速原型和 REST API。


## 练习


<!-- Converted from: 93_Python Flask综合实战.html -->
