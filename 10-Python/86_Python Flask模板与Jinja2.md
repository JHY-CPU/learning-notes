# Python Flask模板与Jinja2


## 🎨 Flask 模板与 Jinja2


Jinja2 模板引擎语法（变量/过滤器/控制结构）、模板继承、静态文件处理、WTForms 表单定义与验证、flash 消息。


## Jinja2 模板语法


```
// ========== Jinja2 基础语法 ==========
// 分隔符:
// {{ ... }} — 表达式 (变量输出)
// {% ... %} — 控制语句 (if/for/block)
// {# ... #} — 注释

// ========== 传参到模板 ==========
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",
        title="首页",
        username="Alice",
        items=["苹果", "香蕉", "橘子"]
    )
```


```
// ========== templates/index.html ==========
// <!DOCTYPE html>
// <html>
// <head>
//     <title>{{ title }}</title>
// </head>
// <body>
//     <h1>欢迎, {{ username }}</h1>
//
//     {# 条件判断 #}
//     {% if items %}
//         <ul>
//         {% for item in items %}
//             <li>{{ item }}</li>
//         {% endfor %}
//         </ul>
//     {% else %}
//         <p>列表为空</p>
//     {% endif %}
// </body>
// </html>
```


## Jinja2 过滤器


```
// ========== 内置过滤器 ==========
// 语法: {{ variable | filter }}
// 链式: {{ variable | filter1 | filter2 }}

// 字符串过滤器:
// {{ "hello" | upper }}          → HELLO
// {{ "HELLO" | lower }}          → hello
// {{ "hello world" | title }}    → Hello World
// {{ " hello " | trim }}         → hello
// {{ "hello" | reverse }}        → olleh

// 列表过滤器:
// {{ [1, 2, 3] | first }}        → 1
// {{ [1, 2, 3] | last }}         → 3
// {{ [3, 1, 2] | sort }}         → [1, 2, 3]
// {{ [1, 2, 3] | length }}       → 3
// {{ [1, 2, 3] | join(", ") }}   → "1, 2, 3"
// {{ [1, 1, 2] | unique }}       → [1, 2]

// 数字过滤器:
// {{ 3.14159 | round(2) }}        → 3.14
// {{ 42 | abs }}                  → 42

// 默认值:
// {{ name | default("游客") }}

// HTML 转义 (安全):
// {{ "
```


<!-- Converted from: 86_Python Flask模板与Jinja2.html -->
