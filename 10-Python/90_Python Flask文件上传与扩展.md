# Python Flask文件上传与扩展


## 📎 Flask 文件上传与扩展


Flask 文件上传处理 (验证/保存/类型检查)、flask-mail 邮件发送、flask-admin 后台管理、flask-caching 缓存、Flask 扩展生态概览。


## 文件上传


```
// ========== 文件上传配置 ==========
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# 确保上传目录存在:
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "没有文件"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    if file and allowed_file(file.filename):
        # secure_filename 清理文件名 (防路径遍历)
        filename = secure_filename(file.filename)
        # 添加时间戳防重名:
        import time
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        return jsonify({
            "message": "上传成功",
            "filename": filename,
            "url": f"/uploads/{filename}"
        }), 201

    return jsonify({"error": "不支持的文件类型"}), 400
```


## 多文件上传与图片处理


```
// ========== 多文件上传 ==========
@app.route("/upload-multiple", methods=["POST"])
def upload_multiple():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "请选择文件"}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            results.append({"filename": filename, "status": "ok"})
        else:
            results.append({"filename": file.filename, "status": "跳过"})

    return jsonify({"results": results})

// ========== 图片处理 (Pillow) ==========
# pip install Pillow

from PIL import Image

def process_image(filepath):
    """压缩图片到指定尺寸"""
    img = Image.open(filepath)
    img.thumbnail((800, 800))        # 限制最大尺寸
    img.save(filepath, optimize=True, quality=85)

    # 创建缩略图:
    thumb = img.copy()
    thumb.thumbnail((200, 200))
    thumb.save(filepath.replace(".", "_thumb."))

@app.route("/upload-image", methods=["POST"])
def upload_image():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    process_image(filepath)  # 处理图片
    return jsonify({"message": "图片上传并处理完成"})
```


## Flask-Mail 邮件


```
// ========== Flask-Mail ==========
# pip install flask-mail

from flask_mail import Mail, Message
from threading import Thread

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "your@email.com"
app.config["MAIL_PASSWORD"] = "your-password"
app.config["MAIL_DEFAULT_SENDER"] = "noreply@example.com"

mail = Mail(app)

# 发送简单邮件:
@app.route("/send-test")
def send_test():
    msg = Message("测试邮件", recipients=["user@example.com"])
    msg.body = "这是一封测试邮件"
    msg.html = "测试这是 HTML 内容"
    mail.send(msg)
    return "邮件已发送"

# 异步发送 (不阻塞请求):
def send_async_email(app, msg):
    with app.app_context():
        mail.send(msg)

@app.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()
    msg = Message(
        subject=f"联系表单: {data['subject']}",
        recipients=["admin@example.com"],
        body=f"来自: {data['email']}\n\n{data['message']}"
    )
    Thread(target=send_async_email, args=(app, msg)).start()
    return jsonify({"message": "邮件已发送"})
```


## Flask-Admin 后台管理


```
// ========== Flask-Admin ==========
# pip install flask-admin

from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

admin = Admin(app, name="管理后台", template_mode="bootstrap4")

# 添加模型管理:
admin.add_view(ModelView(User, db.session))
admin.add_view(ModelView(Post, db.session))

# 自定义管理视图:
from flask_admin import BaseView, expose

class AnalyticsView(BaseView):
    @expose("/")
    def index(self):
        return self.render("admin/analytics.html",
            user_count=User.query.count(),
            post_count=Post.query.count()
        )

admin.add_view(AnalyticsView(name="统计", endpoint="analytics"))

# 访问: http://localhost:5000/admin/
```


## Flask 扩展生态


```
// ========== 常用 Flask 扩展 ==========
// 认证:
// Flask-Login          — Session 用户认证
// Flask-JWT-Extended   — JWT 认证 (API)
// Flask-Security       — 完整认证 + 角色管理

// 数据库:
// Flask-SQLAlchemy     — ORM
// Flask-Migrate        — 数据库迁移
// Flask-MongoEngine    — MongoDB ODM

// 表单与验证:
// Flask-WTF            — WTForms 集成
// Marshmallow          — 序列化/验证

// 缓存:
// Flask-Caching        — Redis/Memcached 缓存
// Flask-Redis          — Redis 集成

// 后台任务:
// Flask-Celery         — Celery 异步任务 (pip install celery)

// 邮件:
// Flask-Mail           — SMTP 邮件发送

// 安全:
// Flask-CORS           — 跨域 (pip install flask-cors)
// Flask-Talisman       — 安全头 (pip install flask-talisman)
// Flask-Limiter        — 限流 (pip install flask-limiter)

// API 文档:
// Flask-RESTx          — Swagger + REST (pip install flask-restx)

// 测试:
// pytest               — 测试框架
```


> **Note:** 💡 文件上传要点: (1) secure_filename 防路径遍历; (2) allowed_file 白名单检查扩展名; (3) MAX_CONTENT_LENGTH 限制文件大小; (4) request.files.getlist() 处理多文件; (5) Pillow 压缩图片/创建缩略图。


## 练习


<!-- Converted from: 90_Python Flask文件上传与扩展.html -->
