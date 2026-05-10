# 项目实战 Express + PostgreSQL REST API


## 📦 项目实战 1: Express + PostgreSQL REST API


完整项目: 目录结构、数据库迁移、CRUD 路由、JWT 认证、Swagger 文档。


## 项目结构


```
// myapp/
// ├── src/
// │   ├── config/        # 配置 (env)
// │   │   └── index.js
// │   ├── db/            # 数据库
// │   │   ├── migrations/   # knex 迁移
// │   │   └── index.js      # 连接池
// │   ├── middleware/    # 中间件
// │   │   ├── auth.js       # JWT 认证
// │   │   ├── validate.js   # 请求验证
// │   │   └── error.js      # 错误处理
// │   ├── routes/        # 路由
// │   │   ├── auth.js
// │   │   └── todos.js
// │   ├── controllers/   # 控制器
// │   │   ├── authController.js
// │   │   └── todoController.js
// │   ├── models/        # 数据模型
// │   │   └── index.js
// │   └── app.js         # Express 入口
// ├── tests/
// ├── .env
// ├── docker-compose.yml
// └── package.json

// ========== 数据库 Schema ==========
// -- migrations/001_create_tables.js
// exports.up = function(knex) {
//   return knex.schema
//     .createTable('users', table => {
//       table.increments('id');
//       table.string('name').notNullable();
//       table.string('email').unique().notNullable();
//       table.string('password_hash').notNullable();
//       table.timestamps(true, true);
//     })
//     .createTable('todos', table => {
//       table.increments('id');
//       table.string('title').notNullable();
//       table.boolean('completed').defaultTo(false);
//       table.integer('user_id')
//            .references('id').inTable('users')
//            .onDelete('CASCADE');
//       table.timestamps(true, true);
//     });
// };
```


## 核心代码


```
// ========== 配置 ==========
// config/index.js
require('dotenv').config();

module.exports = {
  db: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT) || 5432,
    database: process.env.DB_NAME || 'myapp',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || '',
  },
  jwt: {
    secret: process.env.JWT_SECRET || 'dev-secret',
    expiresIn: process.env.JWT_EXPIRES_IN || '15m',
  },
  server: {
    port: parseInt(process.env.PORT) || 3000,
  },
};

// ========== 数据库连接池 ==========
// db/index.js
const knex = require('knex');
const config = require('../config');

const db = knex({
  client: 'pg',
  connection: config.db,
  pool: { min: 2, max: 10 },
});

module.exports = db;

// ========== JWT 中间件 ==========
// middleware/auth.js
const jwt = require('jsonwebtoken');
const config = require('../config');

function authenticate(req, res, next) {
  const header = req.headers.authorization;
  if (!header || !header.startsWith('Bearer ')) {
    return res.status(401).json({ error: '未认证' });
  }

  const token = header.split(' ')[1];
  try {
    const decoded = jwt.verify(token, config.jwt.secret);
    req.userId = decoded.userId;
    next();
  } catch (err) {
    return res.status(401).json({ error: '令牌无效或已过期' });
  }
}

module.exports = { authenticate };

// ========== 控制器 ==========
// controllers/todoController.js
const db = require('../db');

exports.list = async (req, res) => {
  const todos = await db('todos')
    .where({ user_id: req.userId })
    .select('id', 'title', 'completed', 'created_at')
    .orderBy('created_at', 'desc');

  res.json({ data: todos });
};

exports.create = async (req, res) => {
  const { title } = req.body;
  const [todo] = await db('todos')
    .insert({ title, user_id: req.userId })
    .returning(['id', 'title', 'completed', 'created_at']);

  res.status(201).json({ data: todo });
};

exports.update = async (req, res) => {
  const { id } = req.params;
  const { title, completed } = req.body;
  const [todo] = await db('todos')
    .where({ id, user_id: req.userId })
    .update({ title, completed, updated_at: db.fn.now() })
    .returning(['id', 'title', 'completed']);

  if (!todo) return res.status(404).json({ error: '未找到' });
  res.json({ data: todo });
};

exports.remove = async (req, res) => {
  const { id } = req.params;
  const deleted = await db('todos').where({ id, user_id: req.userId }).del();

  if (!deleted) return res.status(404).json({ error: '未找到' });
  res.status(204).send();
};
```


## 路由与应用入口


```
// ========== 路由 ==========
// routes/todos.js
const { Router } = require('express');
const { authenticate } = require('../middleware/auth');
const todoController = require('../controllers/todoController');

const router = Router();

router.use(authenticate); // 所有 todo 路由需要认证

router.get('/',    todoController.list);
router.post('/',   todoController.create);
router.put('/:id', todoController.update);
router.delete('/:id', todoController.remove);

module.exports = router;

// ========== 应用入口 ==========
// app.js
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');

const config = require('./config');
const authRoutes = require('./routes/auth');
const todoRoutes = require('./routes/todos');
const errorHandler = require('./middleware/error');

const app = express();

// 全局中间件
app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN }));
app.use(morgan('combined'));
app.use(express.json());

// 路由
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/todos', todoRoutes);

// 健康检查
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date() });
});

// 统一错误处理
app.use(errorHandler);

// 启动
app.listen(config.server.port, () => {
  console.log(`Server running on port ${config.server.port}`);
});

module.exports = app;

// ========== Docker Compose ==========
// docker-compose.yml:
// version: '3.8'
// services:
//   api:
//     build: .
//     ports:
//       - "3000:3000"
//     environment:
//       DB_HOST: db
//       DB_PASSWORD: secret
//     depends_on:
//       db:
//         condition: service_healthy
//
//   db:
//     image: postgres:16-alpine
//     environment:
//       POSTGRES_PASSWORD: secret
//       POSTGRES_DB: myapp
//     volumes:
//       - pgdata:/var/lib/postgresql/data
//     healthcheck:
//       test: ["CMD-SHELL", "pg_isready -U postgres"]
//       interval: 5s
//
// volumes:
//   pgdata:
```


> **Note:** 💡 Express + PostgreSQL 项目要点: knex 迁移管理 schema; JWT 认证中间件; 分层架构 (routes/controllers/models); 统一错误处理; Docker Compose 本地开发; CRUD RESTful API; .env 多环境配置; helmet/cors/morgan 生产中间件。


## 练习


<!-- Converted from: 0_项目实战 Express  PostgreSQL REST API.html -->
