# Python FastAPI自动文档


## 📖 FastAPI 自动文档


OpenAPI 规范配置、Swagger UI 和 ReDoc 定制、标签 Tags 分组、描述信息 (summary/description/response_description)、OpenAPI 元数据。


## OpenAPI 元数据


```
// ========== 全局元数据配置 ==========
from fastapi import FastAPI

app = FastAPI(
    title="我的 API",
    description="这是一个示例 API 文档\n\n## 功能\n\n- 用户管理\n- 商品管理",
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API 支持",
        "url": "http://example.com/contact",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    # 文档 URL:
    docs_url="/docs",          # Swagger UI (默认)
    redoc_url="/redoc",        # ReDoc (默认)
    openapi_url="/openapi.json", # OpenAPI 规范
)

# 也可以关闭文档 (生产环境):
# app = FastAPI(docs_url=None, redoc_url=None)
```


## Tags 标签分组


```
// ========== 标签分组 ==========
from fastapi import FastAPI, APIRouter

app = FastAPI()

# 方式 1: 在路由中指定 tags
@app.get("/users", tags=["users"])
def list_users():
    """获取用户列表"""
    return [{"id": 1, "name": "Alice"}]

@app.post("/users", tags=["users"])
def create_user():
    """创建用户"""
    return {"message": "created"}

@app.get("/items", tags=["items"])
def list_items():
    """获取商品列表"""
    return [{"id": 1, "name": "键盘"}]

# 方式 2: 用 APIRouter 分组
from fastapi import APIRouter

users_router = APIRouter(prefix="/users", tags=["users"])
items_router = APIRouter(prefix="/items", tags=["items"])

@users_router.get("/")
def get_users():
    return [{"name": "Alice"}]

@users_router.post("/")
def create_user():
    return {"message": "created"}

@items_router.get("/")
def get_items():
    return [{"name": "键盘"}]

app.include_router(users_router)
app.include_router(items_router)

# 标签顺序 (3.0+):
# 在 OpenAPI 中控制标签显示顺序
from fastapi import FastAPI

tags_metadata = [
    {
        "name": "users",
        "description": "用户管理: 注册、登录、个人信息",
        "externalDocs": {
            "description": "外部文档",
            "url": "https://example.com/user-docs/"
        }
    },
    {
        "name": "items",
        "description": "商品管理: 查询、创建、更新、删除",
    },
]

app = FastAPI(openapi_tags=tags_metadata)
```


## 路由文档定制


```
// ========== 路由文档 ==========
from fastapi import FastAPI, Path, Query
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str = Field(..., example="键盘", description="商品名称")
    price: float = Field(..., example=299.0, description="价格")
    tax: float = Field(0.1, example=0.15, description="税率")

@app.post(
    "/items",
    summary="创建商品",              # 简短标题
    description="创建新商品,包含名称、价格和税率",  # 详细描述
    response_description="创建的商品",  # 响应描述
    tags=["items"],
    status_code=201,
)
async def create_item(item: Item):
    """这个 docstring 也会出现在文档中 (Markdown 支持)"""
    return item

# ========== 弃用路由 ==========
@app.get("/old-endpoint", deprecated=True)
def old_endpoint():
    """标记为已弃用 (在文档中显示删除线)"""
    pass

# ========== 隐藏路由 ==========
@app.get("/internal", include_in_schema=False)
def internal():
    """不会出现在 OpenAPI 文档中"""
    pass

# ========== 响应示例 ==========
from fastapi import status
from pydantic import BaseModel

class UserOut(BaseModel):
    id: int = Field(..., example=1)
    username: str = Field(..., example="alice")
    email: str = Field(..., example="alice@example.com")

@app.get(
    "/users/{user_id}",
    response_model=UserOut,
    responses={
        404: {
            "description": "用户不存在",
            "content": {
                "application/json": {
                    "example": {"detail": "User not found"}
                }
            }
        },
        403: {
            "description": "无权限",
            "content": {
                "application/json": {
                    "example": {"detail": "Not authorized"}
                }
            }
        }
    }
)
def get_user(user_id: int):
    if user_id == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": 1, "username": "alice", "email": "alice@example.com"}
```


## 文档定制与安全


```
// ========== Swagger UI 定制 ==========
from fastapi import FastAPI

app = FastAPI(
    swagger_ui_parameters={
        "docExpansion": "none",      # 折叠所有路由
        "defaultModelsExpandDepth": -1,  # 隐藏模型区
        "displayRequestDuration": True,   # 显示请求耗时
        "filter": True,                   # 启用搜索过滤
        "tryItOutEnabled": True,          # 默认启用 Try it out
    }
)

# ========== 安全方案文档 ==========
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

app = FastAPI()

# 自动在文档中添加 Authorize 按钮
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    # 文档中显示锁图标,需要 Authorize
    return {"token": token}

# ========== OpenAPI 自定义 ==========
# 自定义 OpenAPI 函数:
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="自定义 API",
        version="2.0.0",
        description="这是自定义的 OpenAPI 规范",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```


> **Note:** 💡 自动文档要点: (1) FastAPI(title=..., description=..., version=...) 配置元数据; (2) tags=["name"] 分组路由,openapi_tags 控制顺序; (3) summary/description/response_description 文档文本; (4) deprecated=True 标记弃用,include_in_schema=False 隐藏; (5) Swagger UI 的 try it out 可直接测试 API。


## 练习


<!-- Converted from: 98_Python FastAPI自动文档.html -->
