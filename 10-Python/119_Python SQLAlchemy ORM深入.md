# Python SQLAlchemy ORM深入


## 🗃️ SQLAlchemy ORM 深入


关系映射 (一对多/多对多/一对一)、级联操作、懒加载 vs 急加载、joinedload/subqueryload、会话管理、事务控制。


## 模型关系映射


```
// ========== 一对多 ==========
from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)

    # 关系: User → Post (一对多)
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))

    # 反向引用
    author = relationship("User", back_populates="posts")

# ========== 多对多 ==========
from sqlalchemy import Table, Column, Integer, ForeignKey

# 关联表
post_tags = Table(
    "post_tags", Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String(30), unique=True, nullable=False)

    # 多对多关系
    posts = relationship("Post", secondary=post_tags, back_populates="tags")

# 在 Post 中添加:
# tags = relationship("Tag", secondary=post_tags, back_populates="posts")

# ========== 一对一 ==========
class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True)
    bio = Column(Text)
    avatar_url = Column(String(200))
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    # uselist=False 表示一对一
    user = relationship("User", back_populates="profile", uselist=False)

# 在 User 中添加:
# profile = relationship("Profile", back_populates="user", uselist=False)
```


## 级联操作


```
// ========== cascade ==========
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Author(Base):
    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    name = Column(String(50))

    # cascade 选项:
    # save-update: 添加父对象时自动添加子对象
    # delete: 删除父对象时删除所有子对象
    # delete-orphan: 从父对象移除的子对象自动删除
    # all: save-update + merge + refresh-expire + expunge + delete
    # all, delete-orphan: 常用组合 (完全拥有关系)

    books = relationship(
        "Book",
        back_populates="author",
        cascade="all, delete-orphan",
        passive_deletes=True  # 使用数据库级联删除
    )

class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True)
    title = Column(String(100))
    author_id = Column(Integer, ForeignKey("authors.id", ondelete="CASCADE"))

    author = relationship("Author", back_populates="books")

# 使用:
session = Session()

# cascade="all, delete-orphan" 效果:
author = Author(name="张三")
author.books.append(Book(title="书1"))
author.books.append(Book(title="书2"))
session.add(author)    # 自动添加 books
session.commit()

# 删除作者 → 自动删除所有书籍
session.delete(author)
session.commit()
```


## 加载策略


```
// ========== 懒加载 vs 急加载 ==========
from sqlalchemy.orm import joinedload, subqueryload, selectinload, lazyload

# 懒加载 (lazy=True, 默认):
# 访问关系时才查询 (N+1 问题!)
user = session.get(User, 1)
print(user.posts)  # 此时才执行 SELECT * FROM posts WHERE user_id = 1

# N+1 问题:
users = session.query(User).all()
for user in users:  # 1 次查询所有 users
    print(user.posts)  # N 次查询 posts → N+1 次查询!

# ========== 解决方案: 急加载 ==========

# 1. joinedload (JOIN 查询,一次查询)
from sqlalchemy.orm import joinedload

users = session.query(User).options(
    joinedload(User.posts)
).all()
# SQL: SELECT * FROM users LEFT JOIN posts ON users.id = posts.user_id

# 2. subqueryload (子查询)
from sqlalchemy.orm import subqueryload

users = session.query(User).options(
    subqueryload(User.posts)
).all()
# SQL: SELECT * FROM users;
#      SELECT * FROM posts WHERE user_id IN (SELECT id FROM users)

# 3. selectinload (IN 查询)
from sqlalchemy.orm import selectinload

users = session.query(User).options(
    selectinload(User.posts)
).all()
# SQL: SELECT * FROM users;
#      SELECT * FROM posts WHERE user_id IN (1, 2, 3, ...)

# ========== 嵌套加载 ==========
# 加载 User → Post → Tag (两层)
users = session.query(User).options(
    joinedload(User.posts).subqueryload(Post.tags)
).all()

# ========== 只加载特定列 ==========
from sqlalchemy.orm import load_only

users = session.query(User).options(
    load_only(User.name, User.email)  # 只查这两列
).all()
```


## 会话管理


```
// ========== Session 生命周期 ==========
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

engine = create_engine("sqlite:///blog.db")
session = Session(engine)

# 事务控制:
# autoflush=True (默认): 查询前自动 flush
# autocommit=False (默认): 需要手动 commit

# ===== 基本操作 =====
user = User(name="Alice", email="alice@test.com")

# 添加到会话 (INSERT pending)
session.add(user)

# 查看待定的变更
print(session.dirty)    # 已修改的对象
print(session.new)      # 新添加的对象
print(session.deleted)  # 待删除的对象

# 提交事务
session.commit()  # 执行 INSERT,清空待定列表

# 回滚
session.rollback()  # 撤销所有未提交的变更

# ===== 刷新 =====
# flush: 发送 SQL 到数据库 (但不提交)
session.add(user)
session.flush()  # user.id 现在可用了 (如果自增)
print(user.id)   # 可获取 ID

# ===== 关闭 =====
session.close()  # 关闭会话,对象变为 detached 状态

# ===== 使用上下文管理器 =====
with Session(engine) as session:
    user = session.query(User).get(1)
    user.name = "Bob"
    session.commit()  # 自动提交和关闭

# ===== 查询对象状态 =====
from sqlalchemy import inspect

insp = inspect(user)
print(insp.persistent)   # True (在会话中)
print(insp.detached)     # False
print(insp.transient)    # False (从未添加到会话)
print(insp.pending)      # False
print(insp.deleted)      # False
```


## 高级查询


```
// ========== 高级查询 ==========
from sqlalchemy import func, and_, or_, not_, desc, asc, text
from sqlalchemy.orm import aliased

session = Session()

# 聚合查询:
result = session.query(
    User.name,
    func.count(Post.id).label("post_count"),
    func.max(Post.created_at).label("last_post"),
).outerjoin(Post).group_by(User.id).having(
    func.count(Post.id) > 0
).all()

# 子查询:
subq = session.query(Post.user_id, func.count(Post.id).label("cnt")).\
    group_by(Post.user_id).subquery()

users_with_counts = session.query(User, subq.c.cnt).\
    outerjoin(subq, User.id == subq.c.user_id).all()

# UNION:
q1 = session.query(Post.title).filter(Post.id < 10)
q2 = session.query(Post.title).filter(Post.id > 100)
union = q1.union(q2).all()

# 存在查询:
from sqlalchemy import exists
has_posts = session.query(
    exists().where(Post.user_id == User.id)
).scalar()

# 原生 SQL:
result = session.execute(
    text("SELECT id, name FROM users WHERE id > :min_id"),
    {"min_id": 5}
).all()

# 批量插入:
session.bulk_insert_mappings(User, [
    {"name": "A", "email": "a@test.com"},
    {"name": "B", "email": "b@test.com"},
])
session.commit()

# 批量更新:
session.query(User).filter(User.active == True).update(
    {"last_login": datetime.now()},
    synchronize_session="fetch"
)
```


## 完整示例: 博客模型


```
// ========== 完整博客模型 ==========
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, Table, Enum, create_engine, func
)
from sqlalchemy.orm import declarative_base, relationship, Session
from datetime import datetime, UTC
import enum

Base = declarative_base()

# 关联表
post_tags = Table("post_tags", Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class PostStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    bio = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now(UTC), server_default=func.now())

    # 关系
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    profile = relationship("Profile", back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True)
    avatar_url = Column(String(200))
    website = Column(String(200))
    location = Column(String(100))
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    user = relationship("User", back_populates="profile")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False, index=True)
    slug = Column(String(200), unique=True, nullable=False)
    content = Column(Text)
    status = Column(Enum(PostStatus), default=PostStatus.DRAFT)
    views = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now(UTC), server_default=func.now())
    updated_at = Column(DateTime, default=datetime.now(UTC), onupdate=func.now())
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 关系
    author = relationship("User", back_populates="posts")
    tags = relationship("Tag", secondary=post_tags, back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String(30), unique=True, nullable=False)

    posts = relationship("Post", secondary=post_tags, back_populates="posts")

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    author_name = Column(String(50))
    created_at = Column(DateTime, default=datetime.now(UTC))
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False)

    post = relationship("Post", back_populates="comments")
```


> **Note:** 💡 SQLAlchemy 深入要点: relationship 定义关联; cascade 级联删除; joinedload/subqueryload 解决 N+1; 会话管理事务; 聚合/子查询/原生 SQL。


## 练习


<!-- Converted from: 119_Python SQLAlchemy ORM深入.html -->
