# Python Flask基础


## 🌶️ Python Flask 基础


Flask 微框架介绍、应用创建、路由 @app.route、变量规则、HTTP 方法处理、request/response/jsonify、应用运行与调试模式。


## Flask 简介


```
// ========== Flask 是什么 ==========
// Flask: Python 轻量级 Web 框架
// - 微框架 (核心简单,扩展灵活)
// - WSGI 兼容 (Werkzeug + Jinja2)
// - 适合: REST API / 原型 / 中小项目

// 安装:
// pip install flask

// ========== 最小应用 ==========
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)  # 开发服务器

// 运行:
// python app.py
// 访问: http://127.0.0.1:5000
```


## 路由 @app.route


```
// ========== 基本路由 ==========
from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "首页"

@app.route("/about")
def about():
    return "关于页面"

@app.route("/contact")
def contact():
    return "联系方式"

// ========== 变量规则 ==========
@app.route("/user/")
def show_user(username):
    return f"用户: {username}"

@app.route("/post/")
def show_post(post_id):
    return f"文章 ID: {post_id}"

@app.route("/path/")
def show_path(subpath):
    return f"路径: {subpath}"

// 转换器类型:
// string — 默认,接受任何不含 / 的文本
// int    — 整数
// float  — 浮点数
// path   — 接受 / 的路径
// uuid   — UUID 字符串
```


## HTTP 方法


```
// ========== 请求方法 ==========
from flask import Flask, request
app = Flask(__name__)

# 指定方法:
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return "处理登录"
    return "显示登录表单"

# 单一方法快捷方式:
from flask import Flask

@app.get("/users")       # GET 请求
def list_users():
    return "用户列表"

@app.post("/users")      # POST 请求
def create_user():
    return "创建用户"

@app.put("/users/")
def update_user(id):
    return f"更新用户 {id}"

@app.delete("/users/")
def delete_user(id):
    return f"删除用户 {id}"

@app.patch("/users/")
def patch_user(id):
    return f"部分更新用户 {id}"
```


## request 对象


```
// ========== 获取请求数据 ==========
from flask import Flask, request
app = Flask(__name__)

@app.route("/example", methods=["GET", "POST"])
def example():
    # 查询参数: ?name=Alice&age=25
    name = request.args.get("name")        # Alice
    age = request.args.get("age", "18")    # 默认值
    all_params = request.args               # ImmutableMultiDict

    # 表单数据: POST 请求体 (application/x-www-form-urlencoded)
    username = request.form.get("username")

    # JSON 数据: POST 请求体 (application/json)
    data = request.get_json()               # dict 或 None
    if data:
        email = data.get("email")

    # 请求头:
    user_agent = request.headers.get("User-Agent")
    content_type = request.content_type
    auth = request.headers.get("Authorization")

    # URL 各部分:
    print(request.url)          # 完整 URL
    print(request.path)         # /example
    print(request.host)         # 127.0.0.1:5000
    print(request.remote_addr)  # 客户端 IP

    # 上传文件:
    file = request.files.get("file")
    if file:
        file.save(f"uploads/{file.filename}")

    return "OK"
```


## response 对象


```
// ========== 构建响应 ==========
from flask import Flask, jsonify, make_response, redirect, abort, send_file
app = Flask(__name__)

# 方式 1: 直接返回字符串
@app.route("/")
def index():
    return "Hello, World!"

# 方式 2: 返回元组 (body, status_code, headers)
@app.route("/created")
def created():
    return "创建成功", 201, {"X-Custom": "value"}

# 方式 3: make_response
@app.route("/custom")
def custom():
    resp = make_response("自定义")
    resp.status_code = 200
    resp.headers["Content-Type"] = "text/html"
    resp.headers["X-Debug"] = "true"
    resp.set_cookie("token", "abc123", max_age=3600)
    return resp

# 方式 4: jsonify (返回 JSON)
@app.route("/api/user")
def api_user():
    return jsonify({
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com"
    })
    # Content-Type: application/json

# 重定向:
@app.route("/old")
def old():
    return redirect("/new")       # 302 重定向
    # return redirect("/new", 301)  # 永久重定向

# 错误:
@app.route("/notfound")
def not_found():
    abort(404, description="资源不存在")

// ========== 错误处理 ==========
@app.errorhandler(404)
def not_found_handler(error):
    return jsonify({"error": "Not Found", "message": str(error)}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal Server Error"}), 500
```


> **Note:** 💡 Flask 基础要点: (1) @app.route() 定义路由,
> 变量规则; (2) @app.get/post/put/delete 方法快捷方式; (3) request.args/form/get_json/headers 获取请求数据; (4) jsonify 返回 JSON, make_response 自定义响应; (5) abort 触发错误,errorhandler 处理错误。


## 练习


<!-- Converted from: 84_Python Flask基础.html -->
