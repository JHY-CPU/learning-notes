# Python FastAPI数据库集成


## 🗄️ FastAPI 数据库集成


SQLAlchemy + FastAPI 集成、数据库会话依赖、CRUD 操作、Alembic 异步迁移、异步数据库 (SQLAlchemy async / asyncpg)、Redis 缓存集成。


## SQLAlchemy 配置


```
// ========== 数据库配置 ==========
# pip install sqlalchemy psycopg2-binary

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 数据库 URL:
SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@localhost/dbname"
# 也可以: "sqlite:///./app.db"

# 创建引擎:
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,               # 连接池大小
    max_overflow=20,            # 最大溢出连接数
    echo=True,                  # SQL 日志 (开发)
)

# 创建会话工厂:
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 模型基类:
Base = declarative_base()
```


## 模型与依赖


```
// ========== 模型定义 ==========
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(120), unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), index=True)
    price = Column(Float)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="items")
```


## CRUD 与依赖注入


```
// ========== CRUD 模块 ==========
# app/crud.py
from sqlalchemy.orm import Session
from app import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(username=user.username, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()

def create_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```


## FastAPI 路径操作


```
// ========== 完整 CRUD 路由 ==========
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app import crud, schemas, models
from app.database import SessionLocal, engine

app = FastAPI()

# 创建表:
@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=engine)

# 数据库依赖:
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ========== 用户路由 ==========
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="邮箱已存在")
    return crud.create_user(db=db, user=user)

@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return db_user

# ========== 商品路由 ==========
@app.post("/users/{user_id}/items/", response_model=schemas.Item)
def create_item(user_id: int, item: schemas.ItemCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return crud.create_item(db=db, item=item, user_id=user_id)

@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items
```


## Pydantic Schema


```
// ========== Schemas ==========
from pydantic import BaseModel
from datetime import datetime

# Item Schemas:
class ItemBase(BaseModel):
    title: str
    price: float

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True  # Pydantic v1
        # from_attributes = True  # Pydantic v2

# User Schemas:
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    items: list[Item] = []

    class Config:
        orm_mode = True

# ========== orm_mode (Pydantic v1) / from_attributes (v2) ==========
# 允许直接从 ORM 模型创建 Pydantic 模型:
# user = User.model_orm(db_user)  # v2
# user = User.from_orm(db_user)   # v1
```


> **Note:** 💡 数据库集成要点: (1) SQLAlchemy engine + SessionLocal 配置; (2) get_db() yield 依赖管理会话; (3) CRUD 模块分离数据操作; (4) Pydantic orm_mode 支持 ORM → Pydantic 转换; (5) Alembic 用于数据库迁移 (flask db 的 SQLAlchemy 版)。


## 练习


<!-- Converted from: 100_Python FastAPI数据库集成.html -->
