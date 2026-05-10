# Python FastAPI基础


## ⚡ FastAPI 基础入门


FastAPI 介绍、安装与运行、第一个应用、ASGI 服务器 Uvicorn、路径操作装饰器、路由顺序、OpenAPI 自动文档。


## FastAPI 简介


```
// ========== FastAPI 是什么 ==========
            // FastAPI: Python 高性能 Web 框架
            // - 基于 Starlette (ASGI) + Pydantic (数据验证)
            // - 自动生成 OpenAPI 文档 (Swagger/ReDoc)
            // - 内置异步支持 (async/await)
            // - 类型提示驱动 (自动请求验证)
            // - 性能接近 Node.js 和 Go (Starlette + Uvicorn)

            // 适合: REST API / 微服务 / 需要高性能异步场景

            // 安装:
            // pip install fastapi
            // pip install "uvicorn[standard]" # ASGI 服务器

            // ========== 最小应用 ==========
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/")
            def read_root():
            return {"message": "Hello, World!"}

            @app.get("/ping")
            def ping():
            return {"ping": "pong"}

            // 运行:
            // uvicorn main:app --reload
            // uvicorn main:app --host 0.0.0.0 --port 8000

            // ========== 生产环境部署 ==========
            # 开发环境:
            uvicorn main:app --reload

            # 生产环境 (推荐 Gunicorn + Uvicorn workers):
            # pip install gunicorn
            gunicorn main:app \
            -w 4 \
            -k uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:8000 \
            --timeout 120 \
            --keep-alive 5

            # Docker 中:
            # CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]

            // 访问:
            // http://127.0.0.1:8000 → {"message":"Hello, World!"}
            // http://127.0.0.1:8000/docs → Swagger UI (自动!)
            // http://127.0.0.1:8000/redoc → ReDoc UI (自动!)
```


## Uvicorn 运行


```
// ========== Uvicorn 使用 ==========

            # 基本运行:
            uvicorn main:app --reload
            # main: 文件名 (main.py)
            # app: FastAPI 实例变量名
            # --reload: 热重载 (开发用)

            # 生产运行:
            uvicorn main:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers 4 \
            --log-level info \
            --access-log

            # 程序内启动:
            if __name__ == "__main__":
            import uvicorn
            uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

            # ========== 路径操作装饰器 ==========
            @app.get("/") # GET
            @app.post("/") # POST
            @app.put("/") # PUT
            @app.patch("/") # PATCH
            @app.delete("/") # DELETE
            @app.options("/") # OPTIONS
            @app.head("/") # HEAD

            # ========== 路由顺序 ==========
            # ⚠️ 重要: 路由按定义顺序匹配,先定义的优先!
            # 具体路径必须放在动态路径之前

            @app.get("/users/me") # ✅ 必须先定义具体路径
            def get_current_user():
            return {"user": "current"}

            @app.get("/users/{user_id}") # ❌ 如果放在前面, /users/me 会被这个捕获
            def get_user(user_id: int): # user_id 会是字符串 "me",导致验证失败
            return {"user_id": user_id}

            # 错误示例 (会导致 /users/me 无法访问):
            # @app.get("/users/{user_id}") # ❌ 太宽泛,会捕获所有 /users/xxx
            # def get_user(user_id: str):
            # return {"user_id": user_id}
            #
            # @app.get("/users/me") # ❌ 永远不会被匹配到
            # def get_current_user():
            # return {"user": "current"}
```


## FastAPI vs Flask


```
// ========== 对比 ==========
            // 特性 FastAPI Flask
            // 类型 ASGI WSGI
            // 性能 高 (异步) 中 (同步)
            // 自动文档 ✅ OpenAPI/Swagger ❌ (需扩展)
            // 类型提示 ✅ 核心功能 ❌ 可选
            // 异步原生 ✅ async/await ❌ (插件)
            // 数据验证 Pydantic (内置) Marshmallow (扩展)
            // 学习曲线 中等 平缓
            // 社区 快速增长 成熟稳定
            // 适用场景 API/微服务/高性能 通用 Web/中小项目

            // ========== 何时选 FastAPI ==========
            // ✅ JSON API 为主
            // ✅ 需要自动文档
            // ✅ 异步数据库/外部请求
            // ✅ 高性能需求
            // ✅ 微服务架构

            // ========== 何时选 Flask ==========
            // ✅ 需要模板渲染 (Jinja2)
            // ✅ 原型快速开发
            // ✅ 简单应用
            // ✅ 成熟生态
            // ✅ WTForms 表单

            // 两者对比代码:
            # Flask:
            from flask import Flask, jsonify
            app = Flask(__name__)
            @app.route("/")
            def index():
            return jsonify({"hello": "world"})

            # FastAPI:
            from fastapi import FastAPI
            app = FastAPI()
            @app.get("/")
            def index():
            return {"hello": "world"} # 自动 JSON
```


> **Note:** 💡 FastAPI 基础要点: (1) pip install fastapi + uvicorn; (2) uvicorn main:app --reload 启动开发服务器; (3)
>             @app.get/post/put/delete 路径操作; (4) 自动生成 /docs (Swagger) 和 /redoc; (5) 类型提示驱动,返回 dict 自动转 JSON。


## 练习


<!-- Converted from: 94_Python FastAPI基础.html -->
