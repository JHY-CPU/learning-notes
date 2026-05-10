# 项目实战 FastAPI + MongoDB


## 📦 项目实战 2: FastAPI + MongoDB


FastAPI 项目结构、Beanie ODM 模型、异步 CRUD 端点、Pydantic 验证、Docker Compose 开发环境。


## 项目结构


```
# fastapi-mongo/
# ├── app/
# │   ├── __init__.py
# │   ├── main.py             # FastAPI 入口
# │   ├── config.py           # 配置
# │   ├── models/             # Beanie 文档模型
# │   │   ├── __init__.py
# │   │   ├── user.py
# │   │   └── product.py
# │   ├── schemas/            # Pydantic 模型
# │   │   ├── __init__.py
# │   │   ├── user.py
# │   │   └── product.py
# │   ├── routers/            # 路由
# │   │   ├── __init__.py
# │   │   ├── users.py
# │   │   └── products.py
# │   ├── services/           # 业务逻辑
# │   │   ├── __init__.py
# │   │   ├── user_service.py
# │   │   └── product_service.py
# │   └── dependencies.py     # 依赖注入
# ├── tests/
# ├── Dockerfile
# ├── docker-compose.yml
# └── requirements.txt

# ========== 配置 ==========
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "fastapi_mongo"
    secret_key: str = "dev-secret"
    jwt_expires_in: int = 900  # 15分钟

    class Config:
        env_file = ".env"

settings = Settings()

# ========== Beanie ODM 模型 ==========
# models/product.py
from beanie import Document, Indexed
from datetime import datetime
from typing import Optional

class Product(Document):
    name: str
    description: str
    price: float
    category: str
    tags: list[str] = []
    stock: int = 0
    created_at: datetime = datetime.now()
    updated_at: Optional[datetime] = None

    class Settings:
        name = "products"  # collection 名
        indexes = [
            "category",
            [("price", 1)],  # 升序索引
            [("name", "text"), ("description", "text")],  # 全文搜索
        ]
```


## Pydantic Schema 与路由


```
# ========== Pydantic Schema ==========
# schemas/product.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    price: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)
    tags: list[str] = []
    stock: int = Field(0, ge=0)

class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    stock: Optional[int] = Field(None, ge=0)

class ProductResponse(BaseModel):
    id: str
    name: str
    description: str
    price: float
    category: str
    tags: list[str]
    stock: int
    created_at: datetime
    updated_at: Optional[datetime]

# ========== 路由 ==========
# routers/products.py
from fastapi import APIRouter, HTTPException, status
from app.models.product import Product
from app.schemas.product import ProductCreate, ProductUpdate, ProductResponse

router = APIRouter(prefix="/products", tags=["products"])

@router.get("/", response_model=list[ProductResponse])
async def list_products(
    category: str | None = None,
    skip: int = 0,
    limit: int = 20,
):
    query = Product.find()
    if category:
        query = query.where(Product.category == category)

    products = await query.sort(-Product.created_at).skip(skip).limit(limit).to_list()
    return products

@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str):
    product = await Product.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@router.post("/", response_model=ProductResponse, status_code=201)
async def create_product(data: ProductCreate):
    product = Product(**data.model_dump())
    await product.insert()
    return product

@router.put("/{product_id}", response_model=ProductResponse)
async def update_product(product_id: str, data: ProductUpdate):
    product = await Product.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = data.model_dump(exclude_unset=True)
    if update_data:
        update_data["updated_at"] = datetime.now()
        await product.set(update_data)

    return product

@router.delete("/{product_id}", status_code=204)
async def delete_product(product_id: str):
    product = await Product.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    await product.delete()
```


## 应用入口与 Docker


```
# ========== 应用入口 ==========
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings
from app.models.product import Product
from app.models.user import User
from app.routers import products, users

app = FastAPI(
    title="FastAPI MongoDB API",
    description="Product & User management API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(products.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    client = AsyncIOMotorClient(settings.mongodb_url)
    await init_beanie(
        database=client[settings.database_name],
        document_models=[Product, User],
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# ========== Docker Compose ==========
# docker-compose.yml:
# version: '3.8'
# services:
#   api:
#     build: .
#     ports:
#       - "8000:8000"
#     environment:
#       MONGODB_URL: mongodb://mongo:27017
#     depends_on:
#       mongo:
#         condition: service_healthy
#
#   mongo:
#     image: mongo:7
#     volumes:
#       - mongodata:/data/db
#     healthcheck:
#       test: echo 'db.runCommand("ping").ok' | mongosh
#       interval: 10s
#       retries: 5
#
# volumes:
#   mongodata:

# ========== Requirements ==========
# requirements.txt:
# fastapi==0.109.0
# uvicorn[standard]==0.27.0
# beanie==1.23.0
# pydantic-settings==2.1.0
# python-multipart==0.0.6
# bcrypt==4.1.2
# pyjwt==2.8.0

# 启动:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


> **Note:** 💡 FastAPI + MongoDB 项目要点: Beanie ODM 异步驱动; Pydantic 请求/响应验证; FastAPI 自动生成 Swagger 文档; Motor + Beanie 异步 MongoDB; Docker Compose 编排; 索引优化查询; 类型提示自动验证。


## 练习


<!-- Converted from: 1_项目实战 FastAPI  MongoDB.html -->
