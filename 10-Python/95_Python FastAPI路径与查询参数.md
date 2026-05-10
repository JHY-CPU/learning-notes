# Python FastAPI路径与查询参数


## 🔗 FastAPI 路径参数与查询参数


路径参数类型验证、查询参数默认值/可选参数、参数验证 (Path/Query)、字符串验证 (min_length/max_length/regex)、数字验证 (ge/le/gt/lt)。


## 路径参数


```
// ========== 路径参数 ==========
from fastapi import FastAPI

app = FastAPI()

# 基本路径参数:
@app.get("/items/{item_id}")
def read_item(item_id: int):        # 类型注解 → 自动验证!
    return {"item_id": item_id}

# 字符串路径参数:
@app.get("/users/{username}")
def read_user(username: str):
    return {"username": username}

# 枚举路径参数:
from enum import Enum

class UserType(str, Enum):
    admin = "admin"
    user = "user"
    guest = "guest"

@app.get("/users/{user_type}")
def get_users(user_type: UserType):
    # 自动验证: 只接受 admin/user/guest
    return {"type": user_type.value}

# 路径参数包含路径 (FilePath):
@app.get("/files/{file_path:path}")
def read_file(file_path: str):
    return {"file_path": file_path}
    # 访问: /files/home/user/data.txt
    # file_path = "home/user/data.txt"

# 多个路径参数:
@app.get("/users/{user_id}/items/{item_id}")
def get_user_item(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}
```


## 查询参数


```
// ========== 查询参数 ==========
from fastapi import FastAPI

app = FastAPI()

fake_db = [{"id": i, "name": f"Item {i}"} for i in range(100)]

# 查询参数 = 不在路径中的函数参数:
@app.get("/items")
def list_items(
    skip: int = 0,          # 默认值 → 查询参数
    limit: int = 10         # 默认值 → 查询参数
):
    return fake_db[skip:skip + limit]

# 可选查询参数 (Optional):
from typing import Optional

@app.get("/items/{item_id}")
def get_item(
    item_id: int,
    q: Optional[str] = None,  # 可选查询参数
    short: bool = False       # 布尔查询参数
):
    item = {"item_id": item_id}
    if q:
        item["q"] = q
    if not short:
        item["description"] = "完整描述"
    return item

# 必需查询参数 (无默认值):
@app.get("/search")
def search_items(q: str):     # 必须提供 ?q=...
    return {"query": q}

# 布尔查询参数:
@app.get("/items")
def list_items(available: bool = True):
    # URL: /items?available=1 或 ?available=true 或 ?available=True
    return {"available": available}

# 列表查询参数:
from typing import List

@app.get("/items")
def list_items(tags: List[str] = Query([])):
    # URL: /items?tags=a&tags=b&tags=c
    return {"tags": tags}
```


## 参数验证 (Path / Query)


```
// ========== Path & Query 验证 ==========
from fastapi import FastAPI, Path, Query

app = FastAPI()

# Path 验证:
@app.get("/items/{item_id}")
def read_item(
    item_id: int = Path(..., title="文章 ID", ge=1, le=1000),
):
    # ... = 必需
    # ge=1: 大于等于 1
    # le=1000: 小于等于 1000
    return {"item_id": item_id}

# Query 验证:
@app.get("/items")
def read_items(
    q: Optional[str] = Query(
        None,                           # 默认值
        alias="search",                 # URL 中参数名: ?search=...
        title="搜索关键词",
        description="搜索商品的名称",
        min_length=2,                   # 最小长度
        max_length=50,                  # 最大长度
        regex="^[a-zA-Z0-9]+$",        # 正则匹配
        deprecated=True                 # 标记为已弃用
    ),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
):
    return {"q": q, "page": page, "size": size}

# 数字验证:
@app.get("/products/{product_id}")
def read_product(
    product_id: int = Path(..., ge=1),     # ≥ 1
    price: float = Query(..., gt=0),       # > 0
    discount: float = Query(0, ge=0, le=1) # 0-1 之间
):
    return {"product_id": product_id, "price": price}

# 必需查询参数:
@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=100)
):
    # ... 表示必须提供
    return {"query": q}
```


## 参数排序与文档


```
// ========== 参数排序 ==========
# FastAPI 按以下规则判断参数来源:
# 1. 路径参数 → 声明在路径中
# 2. 查询参数 → 其他参数

# 如果参数既是路径又是查询 → 路径优先

# ========== 带文档的 API ==========
@app.get("/users/{user_id}")
def get_user(
    user_id: int = Path(..., title="用户ID", description="用户的唯一标识", example=42),
    include_posts: bool = Query(False, description="是否包含用户文章"),
    limit: int = Query(10, ge=1, le=100, description="文章数量限制")
):
    """获取用户信息。

    根据用户 ID 获取用户详情,可选择包含文章列表。
    """
    return {"user_id": user_id, "include_posts": include_posts}

# 生成的 Swagger 文档会自动包含:
# - 参数名称、类型、是否必需
# - 描述和示例值
# - 验证规则 (min/max/pattern)

# ========== 多个路径和查询 ==========
@app.get("/products/{category}/{product_id}")
def get_product(
    category: str = Path(..., title="分类", regex="^(electronics|clothing|food)$"),
    product_id: int = Path(..., ge=1),
    fields: Optional[str] = Query(None, description="返回字段,逗号分隔"),
    sort: str = Query("name", regex="^(name|price|rating)$")
):
    return {
        "category": category,
        "product_id": product_id,
        "fields": fields,
        "sort": sort
    }
```


> **Note:** 💡 参数要点: (1) 路径参数用类型注解自动验证类型; (2) 有默认值/可选→查询参数,无默认值→必需查询参数; (3) Path/Query 添加验证: ge/le/min_length/regex; (4) ... 表示必需; (5) alias 修改 URL 参数名。


## 练习


<!-- Converted from: 95_Python FastAPI路径与查询参数.html -->
