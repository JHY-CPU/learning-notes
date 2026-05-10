# Python Flask REST API


## 🔌 Flask REST API 构建


使用 Flask 构建 RESTful API、Blueprint 模块化组织、请求验证（marshmallow）、统一错误处理、分页与过滤、API 文档生成。


## RESTful API 设计


```
// ========== API 路由设计 ==========
// 资源: POSTS (文章)
//
// GET    /posts          — 文章列表
// POST   /posts          — 创建文章
// GET    /posts/{id}     — 文章详情
// PUT    /posts/{id}     — 更新文章
// DELETE /posts/{id}     — 删除文章
//
// GET    /posts/{id}/comments — 文章评论

from flask import Flask, jsonify, request, abort
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)

# API 端点:
@app.route("/api/posts", methods=["GET"])
def list_posts():
    posts = Post.query.all()
    return jsonify([{
        "id": p.id,
        "title": p.title,
        "content": p.content
    } for p in posts])
```


## 完整 CRUD 示例


```
// ========== 完整 CRUD ==========
from flask import Blueprint, jsonify, request
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
posts_bp = Blueprint("posts", __name__, url_prefix="/api/posts")

# 列表 + 创建:
@posts_bp.route("", methods=["GET", "POST"])
def posts():
    if request.method == "GET":
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 10, type=int)
        pagination = Post.query.paginate(page=page, per_page=per_page)

        return jsonify({
            "items": [p.to_dict() for p in pagination.items],
            "total": pagination.total,
            "page": page,
            "pages": pagination.pages
        })

    # POST
    data = request.get_json()
    if not data or "title" not in data:
        return jsonify({"error": "title 是必填字段"}), 400

    post = Post(title=data["title"], content=data.get("content", ""))
    db.session.add(post)
    db.session.commit()

    return jsonify(post.to_dict()), 201

# 详情 / 更新 / 删除:
@posts_bp.route("/", methods=["GET", "PUT", "DELETE"])
def post_detail(post_id):
    post = Post.query.get_or_404(post_id)

    if request.method == "GET":
        return jsonify(post.to_dict())

    if request.method == "PUT":
        data = request.get_json()
        post.title = data.get("title", post.title)
        post.content = data.get("content", post.content)
        db.session.commit()
        return jsonify(post.to_dict())

    # DELETE
    db.session.delete(post)
    db.session.commit()
    return jsonify({"message": "删除成功"}), 200

# 模型 to_dict 方法:
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, default="")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content
        }
```


## 请求验证 (marshmallow)


```
// ========== Marshmallow 模式验证 ==========
// 安装: pip install marshmallow

from marshmallow import Schema, fields, validate, ValidationError

class PostSchema(Schema):
    id = fields.Int(dump_only=True)     # 只用于序列化
    title = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    content = fields.Str(missing="")     # 默认为空
    created_at = fields.DateTime(dump_only=True)

class UserSchema(Schema):
    username = fields.Str(required=True, validate=validate.Length(min=3, max=50))
    email = fields.Email(required=True)
    age = fields.Int(validate=validate.Range(min=0, max=150))

# 使用:
post_schema = PostSchema()
posts_schema = PostSchema(many=True)

@posts_bp.route("", methods=["POST"])
def create_post():
    try:
        data = post_schema.load(request.get_json())
    except ValidationError as e:
        return jsonify({"errors": e.messages}), 400

    post = Post(**data)
    db.session.add(post)
    db.session.commit()

    return post_schema.dump(post), 201

@posts_bp.route("", methods=["GET"])
def list_posts():
    posts = Post.query.all()
    return jsonify(posts_schema.dump(posts))
```


## 统一错误处理


```
// ========== 统一错误处理 ==========
from flask import jsonify

class APIError(Exception):
    """自定义 API 错误"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = {"error": self.message}
        if self.payload:
            rv.update(self.payload)
        return rv

# 注册错误处理器:
@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "资源不存在"}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "请求无效"}), 400

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "服务器内部错误"}), 500

# 使用:
@app.route("/api/users/")
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        raise APIError("用户不存在", 404)
    return jsonify(user.to_dict())
```


## API 项目结构


```
// ========== 推荐结构 ==========
# myapi/
# ├── app.py                # 应用工厂
# ├── config.py             # 配置
# ├── models/
# │   ├── __init__.py
# │   ├── user.py
# │   └── post.py
# ├── routes/
# │   ├── __init__.py
# │   ├── users.py
# │   └── posts.py
# ├── schemas/
# │   ├── __init__.py
# │   ├── user_schema.py
# │   └── post_schema.py
# ├── errors.py             # 错误处理
# └── requirements.txt

// ========== app.py ==========
def create_app():
    app = Flask(__name__)
    app.config.from_object("config.Config")

    db.init_app(app)
    migrate.init_app(app, db)

    # 注册蓝图
    from routes.posts import posts_bp
    app.register_blueprint(posts_bp, url_prefix="/api")

    # 错误处理
    from errors import register_handlers
    register_handlers(app)

    return app

// ========== flask-cors: 跨域 ==========
from flask_cors import CORS
CORS(app)  # 允许所有来源

# 或限制:
CORS(app, origins=["https://example.com"])
```


> **Note:** 💡 REST API 要点: (1) RESTful 路由: GET/POST/PUT/DELETE 对应查询/创建/更新/删除; (2) Blueprint 模块化组织路由; (3) marshmallow Schema 验证请求 + 序列化响应; (4) 统一错误处理 raise APIError + @app.errorhandler; (5) to_dict() 模型方法简化 JSON 序列化。


## 练习


<!-- Converted from: 88_Python Flask REST API.html -->
