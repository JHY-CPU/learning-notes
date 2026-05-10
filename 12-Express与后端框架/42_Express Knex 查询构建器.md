# Express Knex 查询构建器


## 🔨 Express Knex 查询构建器


Knex.js 查询构建器、链式查询、数据库迁移 (Migration)、种子数据 (Seed)、事务、多数据库支持、原始查询、聚合与 JOIN、Schema 构建。


## Knex 基础


```
// ========== Knex.js ==========
// SQL 查询构建器 (支持 PostgreSQL/MySQL/SQLite 等)
// 比 raw SQL 更安全, 支持链式调用

// 安装:
// npm install knex
// npm install pg   (PostgreSQL 驱动)

const knex = require('knex');

// ========== 初始化 ==========
const db = knex({
    client: 'pg',
    connection: {
        host: process.env.DB_HOST || 'localhost',
        port: parseInt(process.env.DB_PORT, 10) || 5432,
        database: process.env.DB_NAME || 'myapp',
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASS || 'postgres',
        ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
    },
    pool: {
        min: 2,
        max: 10,
    },
    // 打印 SQL (开发用)
    // log: { warn: console.warn, error: console.error },
});

// ========== 基础 CRUD ==========
// INSERT
const [user] = await db('users')
    .insert({ name: 'Alice', email: 'alice@test.com' })
    .returning('*');  // PostgreSQL 支持 RETURNING

// SELECT (所有)
const users = await db('users').select('*');

// SELECT (特定字段, 条件)
const users = await db('users')
    .select('id', 'name', 'email')
    .where('is_active', true)
    .orderBy('created_at', 'desc')
    .limit(20)
    .offset(0);

// 单条
const user = await db('users')
    .where({ id: 1 })
    .first();  // 取第一条

// UPDATE
const [updated] = await db('users')
    .where({ id: 1 })
    .update({ name: 'Alice Updated', updated_at: db.fn.now() })
    .returning('*');

// DELETE
await db('users')
    .where({ id: 1 })
    .del();

// ========== Express 集成 ==========
// config/database.js
const config = require('./index');
const knex = require('knex');

const db = knex({
    client: config.db.client,     // 'pg' | 'mysql2' | 'sqlite3'
    connection: config.db.uri,
    pool: { min: 2, max: 10 },
});

module.exports = db;

// 路由中使用:
app.get('/users', async (req, res) => {
    const users = await db('users')
        .select('id', 'name', 'email')
        .where({ is_active: true })
        .orderBy('created_at', 'desc');

    res.success(users);
});
```


## 高级查询


```
// ========== 高级查询 ==========

// ========== JOIN ==========
const orders = await db('orders')
    .join('users', 'orders.user_id', 'users.id')
    .leftJoin('order_items', 'orders.id', 'order_items.order_id')
    .select(
        'orders.id',
        'orders.total',
        'users.name as user_name',
        'users.email'
    )
    .where('orders.status', 'completed')
    .groupBy('orders.id');

// ========== 聚合 ==========
const stats = await db('orders')
    .select(
        db.raw("DATE(created_at) as date"),
        db.raw('COUNT(*) as order_count'),
        db.raw('SUM(total) as revenue'),
        db.raw('AVG(total) as avg_order'),
    )
    .where('created_at', '>=', '2024-01-01')
    .groupByRaw('DATE(created_at)')
    .orderBy('date', 'asc');

// ========== 子查询 ==========
const activeUsers = await db('users')
    .whereIn('id', function() {
        this.select('user_id')
            .from('orders')
            .where('created_at', '>', '2024-01-01');
    });

// ========== 条件构建 ==========
function buildUserQuery(filters) {
    let query = db('users').select('*');

    if (filters.search) {
        query = query.where(function() {
            this.where('name', 'ilike', `%${filters.search}%`)
                .orWhere('email', 'ilike', `%${filters.search}%`);
        });
    }

    if (filters.role) {
        query = query.where('role', filters.role);
    }

    if (filters.isActive !== undefined) {
        query = query.where('is_active', filters.isActive);
    }

    if (filters.createdAfter) {
        query = query.where('created_at', '>=', filters.createdAfter);
    }

    query = query
        .orderBy(filters.sort || 'created_at', filters.order || 'desc')
        .limit(filters.limit || 20)
        .offset(((filters.page || 1) - 1) * (filters.limit || 20));

    return query;
}

// ========== 原生 SQL ==========
const result = await db.raw(`
    SELECT id, name,
           ROW_NUMBER() OVER (ORDER BY created_at DESC) as rank
    FROM users
    WHERE is_active = ?
`, [true]);

// ========== 事务 ==========
await db.transaction(async (trx) => {
    const [order] = await trx('orders')
        .insert({ user_id: userId, total: amount })
        .returning('*');

    await trx('order_items')
        .insert(items.map(item => ({ ...item, order_id: order.id })));

    await trx('inventory')
        .where('product_id', productId)
        .decrement('quantity', quantity);
});
```


## 迁移与种子


```
// ========== Knex Migrations ==========
// 数据库版本控制

// 安装 + 初始化:
// npx knex init          → 创建 knexfile.js
// npx knex migrate:make create_users   → 创建迁移文件

// ========== knexfile.js ==========
module.exports = {
    development: {
        client: 'pg',
        connection: {
            host: 'localhost',
            database: 'myapp_dev',
            user: 'postgres',
            password: 'postgres',
        },
        migrations: { directory: './migrations' },
        seeds: { directory: './seeds' },
    },
    production: {
        client: 'pg',
        connection: process.env.DATABASE_URL,
        pool: { min: 2, max: 10 },
        migrations: { directory: './migrations' },
    },
};

// ========== 创建迁移 ==========
// npx knex migrate:make create_users
// → migrations/20240101000000_create_users.js

exports.up = function(knex) {
    return knex.schema.createTable('users', (table) => {
        table.increments('id').primary();
        table.string('name', 255).notNullable();
        table.string('email', 255).unique().notNullable();
        table.string('password_hash', 255).notNullable();
        table.string('role', 50).defaultTo('user');
        table.boolean('is_active').defaultTo(true);
        table.timestamps(true, true);  // created_at + updated_at
    });
};

exports.down = function(knex) {
    return knex.schema.dropTableIfExists('users');
};

// ========== Schema 构建方法 ==========
// 表操作:
// knex.schema.createTable('users', ...)
// knex.schema.dropTable('users')
// knex.schema.renameTable('old', 'new')
// knex.schema.hasTable('users')

// 列操作:
// table.increments('id')       自增主键
// table.string('name', 100)    字符串
// table.text('description')    长文本
// table.integer('age')         整数
// table.float('price', 8, 2)   浮点
// table.boolean('active')      布尔
// table.date('birthday')       日期
// table.datetime('created_at') 日期时间
// table.timestamps()           created_at + updated_at
// table.json('metadata')       JSON
// table.enum('role', ['admin', 'user'])
// table.uuid('id').primary()   UUID

// 约束/索引:
// .notNullable()
// .defaultTo(value)
// .unique()
// .primary()
// .references('users.id')     外键
// .onDelete('CASCADE')
// table.index(['email', 'name'])

// ========== 种子数据 ==========
// npx knex seed:make add_users
// → seeds/01_add_users.js

exports.seed = async function(knex) {
    // 删除现有
    await knex('users').del();

    // 插入种子数据
    await knex('users').insert([
        { name: 'Admin', email: 'admin@test.com', role: 'admin' },
        { name: 'User 1', email: 'user1@test.com', role: 'user' },
        { name: 'User 2', email: 'user2@test.com', role: 'user' },
    ]);
};

// 运行:
// npx knex migrate:latest        # 运行所有迁移
// npx knex migrate:rollback      # 回滚
// npx knex migrate:up            # 运行下一个
// npx knex seed:run              # 运行种子
```


## Knex Express 模式


```
// ========== Knex + Express 模式 ==========

// ========== Model 层 ==========
// models/user.js
const db = require('../config/database');

class UserModel {
    static tableName = 'users';

    static async create(data) {
        const [user] = await db(this.tableName)
            .insert(data)
            .returning(['id', 'name', 'email', 'role', 'created_at']);
        return user;
    }

    static async findById(id) {
        return db(this.tableName)
            .select('id', 'name', 'email', 'role', 'created_at')
            .where({ id })
            .first();
    }

    static async findAll({ page = 1, limit = 20, search, role, sort = 'created_at' }) {
        let query = db(this.tableName)
            .select('id', 'name', 'email', 'role', 'is_active', 'created_at');

        if (search) {
            query = query.where(function() {
                this.where('name', 'ilike', `%${search}%`)
                    .orWhere('email', 'ilike', `%${search}%`);
            });
        }
        if (role) query = query.where('role', role);

        const [{ count }] = await query.clone().count('* as count').first();

        const rows = await query
            .orderBy(sort, 'desc')
            .limit(limit)
            .offset((page - 1) * limit);

        return { rows, total: parseInt(count), page, limit };
    }

    static async update(id, data) {
        const [user] = await db(this.tableName)
            .where({ id })
            .update({ ...data, updated_at: db.fn.now() })
            .returning(['id', 'name', 'email', 'role', 'updated_at']);
        return user;
    }

    static async delete(id) {
        return db(this.tableName).where({ id }).del();
    }
}

// ========== Service 层 ==========
// services/userService.js
class UserService {
    static async register(data) {
        // 检查邮箱唯一
        const existing = await UserModel.findByEmail(data.email);
        if (existing) throw new Error('Email already exists');

        const passwordHash = await bcrypt.hash(data.password, 12);
        return UserModel.create({
            name: data.name,
            email: data.email,
            password_hash: passwordHash,
        });
    }
}

// ========== Knex 多数据库切换 ==========
// 只需改配置:
// client: 'pg'     → PostgreSQL
// client: 'mysql2' → MySQL
// client: 'sqlite3' → SQLite
// client: 'mssql'  → SQL Server

// ========== Knex vs Raw SQL vs ORM ==========
// Raw SQL: 性能最佳, 需手动防注入
// Knex:    SQL 抽象, 链式调用, 迁移
// ORM:     对象映射, 自动关联, 开发快
```


> **Note:** 💡 Knex 要点: 查询构建器链式调用; 自动参数化防注入; 迁移版本控制 (up/down); 种子数据填充; Schema 构建 (table.string/integer/boolean); JOIN/聚合/子查询; 事务 trx 对象; 多数据库切换只改 client; RETURNING 取回数据; 条件查询动态构建; knexfile 环境管理。


## 练习


<!-- Converted from: 42_Express Knex 查询构建器.html -->
