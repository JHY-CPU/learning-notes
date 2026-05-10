# Python FastAPI依赖注入


## 💉 FastAPI 依赖注入


Depends 依赖注入系统：函数依赖/类依赖/可调用依赖、共享数据、依赖嵌套、全局依赖、选择性依赖、带 yield 依赖（资源清理）。


## 依赖注入基础


```
// ========== 什么是依赖注入 ==========
// 依赖注入: 自动提供函数所需的依赖
// 在 FastAPI 中,用 Depends() 声明依赖

from fastapi import FastAPI, Depends

app = FastAPI()

# ========== 最简单的依赖 ==========
# 依赖就是一个可调用对象 (函数/类)

def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    """依赖函数: 提供通用查询参数"""
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items")
def list_items(commons: dict = Depends(common_parameters)):
    # FastAPI 自动调用 common_parameters()
    # 并传入请求中的查询参数
    return commons

@app.get("/users")
def list_users(commons: dict = Depends(common_parameters)):
    # 复用 common_parameters 依赖
    return commons

# 依赖函数的参数也会被 FastAPI 解析
# 依赖可以有子依赖!
```


## 类依赖与可调用


```
// ========== 类依赖 ==========
from fastapi import Depends

class CommonQueryParams:
    """类作为依赖"""
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items")
def list_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    # commons 是 CommonQueryParams 实例
    return {"q": commons.q, "skip": commons.skip}

# 简写 (FastAPI 自动推断):
@app.get("/items")
def list_items(commons: CommonQueryParams = Depends()):
    # Depends() 自动使用类型注解中的类
    return commons

# ========== 多层依赖嵌套 ==========
def query_extractor(q: str | None = None):
    """第一层依赖"""
    return q

def query_or_cookie_extractor(
    q: str = Depends(query_extractor),
    last_query: str | None = None
):
    """第二层依赖,依赖 query_extractor"""
    if not q:
        return last_query
    return q

@app.get("/items")
def read_query(
    query_or_default: str = Depends(query_or_cookie_extractor)
):
    return {"q_or_cookie": query_or_default}

# FastAPI 自动处理依赖链
```


## 带 yield 的依赖


```
// ========== yield 依赖 (资源管理) ==========
# Python 3.7+ 支持: 在 yield 前获取资源
# yield 后清理资源 (类似上下文管理器)

from fastapi import Depends

# 数据库会话依赖:
def get_db():
    """提供数据库会话,请求结束后关闭"""
    db = SessionLocal()      # 创建会话
    try:
        yield db             # 注入到路径操作
    finally:
        db.close()           # 请求结束后关闭

@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

# ========== 多个 yield 依赖 ==========
# FastAPI 按逆序清理 (后入先出)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(db: Session = Depends(get_db)):
    """依赖 get_db, 也使用 yield 清理"""
    user = db.query(User).first()
    try:
        yield user
    finally:
        pass  # 清理逻辑

@app.get("/me")
def read_me(current_user: User = Depends(get_current_user)):
    return current_user

# 清理顺序: get_current_user → get_db
```


## 全局依赖与路径装饰器


```
// ========== 路径装饰器依赖 ==========
from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()

# 只执行不返回值的依赖 (用于验证):
def verify_token(x_token: str = Header(...)):
    if x_token != "secret-token":
        raise HTTPException(status_code=400, detail="无效 Token")

def verify_key(x_key: str = Header(...)):
    if x_key != "secret-key":
        raise HTTPException(status_code=400, detail="无效 Key")
    return x_key  # 可以有返回值

# 路径装饰器中使用 dependencies:
@app.get("/protected", dependencies=[Depends(verify_token), Depends(verify_key)])
def protected_route():
    # 先验证 token 和 key, 通过后才执行
    return {"message": "安全路径"}

# ========== 全局依赖 ==========
# 应用到所有路由:
app = FastAPI(dependencies=[Depends(verify_token)])

# 或:
app = FastAPI()
app.dependencies.append(Depends(verify_token))

# ========== 路径操作中的依赖 ==========
@app.get("/items", dependencies=[Depends(verify_token)])
def get_items():
    return ["item1", "item2"]
```


## Depends 最佳实践


```
// ========== 依赖设计模式 ==========
# 1. 共享逻辑: 认证、日志、分页、数据库会话
# 2. 缓存结果: Depends 默认缓存 (同一个依赖在同一请求中只调用一次)
# 3. 组合使用: 小依赖可组合成大依赖

# ========== 缓存行为 ==========
def expensive_query():
    """同一请求中多次调用也只执行一次"""
    print("查询数据库...")
    return {"data": "expensive result"}

@app.get("/items")
def read_items(a: dict = Depends(expensive_query), b: dict = Depends(expensive_query)):
    # expensive_query 只执行一次!
    return {"a": a, "b": b}

# ========== use_cache 参数 ==========
# use_cache=False 禁止缓存 (每次都重新执行):
@app.get("/no-cache")
def no_cache(a: dict = Depends(expensive_query, use_cache=False)):
    return a

# ========== 依赖 vs 路径操作参数 ==========
# ✅ 认证/授权 → Depends
# ✅ 数据库会话 → Depends
# ✅ 通用分页 → Depends
# ✅ 请求参数 → 直接函数参数
# ✅ 业务逻辑 → 路径操作函数内

# ========== 按模块组织依赖 ==========
# app/dependencies.py:
# def get_db(): ...
# def get_current_user(): ...
# def get_pagination(): ...
# def verify_role(role: str): ...
```


> **Note:** 💡 依赖注入要点: (1) Depends() 自动调用依赖并注入结果; (2) 依赖可以是函数/类/可调用对象; (3) yield 依赖可管理资源生命周期; (4) dependencies=[] 路径装饰器只验证不注入; (5) 同一请求中 Depends 默认缓存结果。


## 练习


<!-- Converted from: 99_Python FastAPI依赖注入.html -->
