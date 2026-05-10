# Express Body 解析与验证


## 📦 Express Body 解析与验证


express.json() 配置 (limit/type/verify)、express.urlencoded 表单解析、raw/text 解析、文件上传 multer、请求体验证 Joi/Zod 模式、验证错误处理中间件。


## Body 解析中间件


```
// ========== Body 解析 ==========
// Express 内置解析器, 处理请求体

// ========== JSON 解析 ==========
app.use(express.json());
// 解析 Content-Type: application/json
// req.body 自动解析为 JS 对象

// 高级配置:
app.use(express.json({
    limit: '1mb',                    // 限制 body 大小 (默认 100kb)
    type: 'application/json',       // 只解析此 Content-Type
    strict: true,                    // 只解析对象和数组 (非字符串)
    inflate: true,                    // 支持压缩
    verify: (req, res, buf) => {
        // 验证原始 body (签名验证等)
        // buf 是 Buffer
        try {
            JSON.parse(buf);
        } catch (e) {
            throw new Error('Invalid JSON');
        }
    }
}));

// ========== URL-encoded 表单 ==========
app.use(express.urlencoded({ extended: true }));
// 解析 Content-Type: application/x-www-form-urlencoded
// extended: true 使用 qs 库 (支持嵌套对象)
// extended: false 使用 queryString 库 (扁平)

// 示例: form 提交 username=alice&age=25
// req.body = { username: "alice", age: "25" }

// ========== raw 解析 ==========
app.use(express.raw({ type: 'application/octet-stream', limit: '10mb' }));
// req.body 为 Buffer

// ========== text 解析 ==========
app.use(express.text({ type: 'text/plain' }));
// req.body 为字符串

// ========== 按路由区分 ==========
// 只在特定路由使用:
app.post('/webhook', express.raw({ type: '*/*' }), (req, res) => {
    const sig = req.get('X-Signature');
    // 验证原始 body Buffer
});
```


## 请求体验证 (Joi)


```
// ========== Joi 验证 ==========
// 最流行的 Node.js 验证库

npm install joi

// ========== Schema 定义 ==========
const Joi = require('joi');

const createUserSchema = Joi.object({
    name: Joi.string()
        .min(2)
        .max(50)
        .required()
        .messages({
            'string.min': 'Name must be at least 2 characters',
            'any.required': 'Name is required',
        }),

    email: Joi.string()
        .email()
        .required(),

    password: Joi.string()
        .min(8)
        .max(128)
        .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
        .message('Password must contain uppercase, lowercase, and number')
        .required(),

    age: Joi.number()
        .integer()
        .min(0)
        .max(150),

    role: Joi.string()
        .valid('user', 'admin', 'moderator')
        .default('user'),

    tags: Joi.array()
        .items(Joi.string())
        .max(10),
});

// ========== 验证中间件 ==========
function validate(schema) {
    return (req, res, next) => {
        const { error, value } = schema.validate(req.body, {
            abortEarly: false,    // 返回所有错误
            stripUnknown: true,    // 移除未定义的字段
        });

        if (error) {
            const errors = error.details.map(detail => ({
                field: detail.path.join('.'),
                message: detail.message,
            }));

            return res.status(422).json({
                code: 'VALIDATION_ERROR',
                message: 'Validation failed',
                errors,
            });
        }

        req.body = value;   // 使用清洗后的数据
        next();
    };
}

// ========== 使用 ==========
app.post('/users', validate(createUserSchema), asyncHandler(async (req, res) => {
    const user = await User.create(req.body);   // req.body 已经验证
    res.status(201).json(user);
}));
```


## 请求体验证 (Zod)


```
// ========== Zod ==========
// TypeScript 优先的验证库

npm install zod

// ========== Schema 定义 ==========
const { z } = require('zod');

const createUserSchema = z.object({
    name: z.string()
        .min(2, 'Name too short')
        .max(50, 'Name too long'),

    email: z.string()
        .email('Invalid email'),

    password: z.string()
        .min(8)
        .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
            'Password must contain uppercase, lowercase, number'),

    age: z.number()
        .int()
        .positive()
        .optional(),

    role: z.enum(['user', 'admin'])
        .default('user'),

    tags: z.array(z.string())
        .max(10)
        .optional(),
});

// ========== 类型推导 ==========
// Zod 的一大优势: 从 schema 推导 TS 类型
type CreateUserDTO = z.infer;

// ========== Zod 验证中间件 ==========
function validateZod(schema) {
    return (req, res, next) => {
        const result = schema.safeParse(req.body);

        if (!result.success) {
            const errors = result.error.issues.map(issue => ({
                field: issue.path.join('.'),
                message: issue.message,
            }));

            return res.status(422).json({
                code: 'VALIDATION_ERROR',
                message: 'Validation failed',
                errors,
            });
        }

        req.body = result.data;
        next();
    };
}

// ========== Joi vs Zod ==========
// ┌──────────┬──────────────────┬──────────────────────┐
// │          │ Joi              │ Zod                  │
// ├──────────┼──────────────────┼──────────────────────┤
// │ TypeScript│ 一般            │ 优秀 (类型推导)      │
// │ 生态     │ 成熟, 插件多    │ 较新, 增长快         │
// │ 错误信息 │ 可定制           │ 可定制               │
// │ 学习曲线 │ 简单             │ 简单                 │
// │ 适合     │ JS 项目          │ TS 项目优先          │
// └──────────┴──────────────────┴──────────────────────┘
```


## 查询/参数验证


```
// ========== 查询参数和路径参数验证 ==========
// 不仅验证 body, 也要验证 query 和 params

// ========== 查询参数验证 ==========
const listUsersSchema = Joi.object({
    page: Joi.number().integer().min(1).default(1),
    limit: Joi.number().integer().min(1).max(100).default(20),
    sort: Joi.string().valid('name', 'createdAt', 'email').default('createdAt'),
    order: Joi.string().valid('asc', 'desc').default('desc'),
    search: Joi.string().allow('').optional(),
});

// 通用验证中间件:
function validateQuery(schema) {
    return (req, res, next) => {
        const { error, value } = schema.validate(req.query, {
            abortEarly: false,
            stripUnknown: true,
        });

        if (error) {
            return res.status(422).json({
                code: 'INVALID_QUERY',
                message: error.details.map(d => d.message).join(', '),
            });
        }

        req.query = value;
        next();
    };
}

app.get('/users', validateQuery(listUsersSchema), asyncHandler(async (req, res) => {
    // req.query 已验证, 含默认值
    const { page, limit, sort, order, search } = req.query;
    // ...
}));

// ========== 路径参数验证 ==========
function validateParams(schema) {
    return (req, res, next) => {
        const { error, value } = schema.validate(req.params);
        if (error) {
            return res.status(400).json({ message: 'Invalid params' });
        }
        req.params = value;
        next();
    };
}

app.get('/users/:id(\\d+)',
    validateParams(Joi.object({ id: Joi.number().integer().required() })),
    asyncHandler(async (req, res) => {
        const { id } = req.params;
    })
);
```


> **Note:** 💡 Body 解析要点: express.json() 默认 100kb 限制; 文件上传用 multer 而非 JSON; Joi/Zod 验证请求体, 提前拦截无效数据; abortEarly: false 返回所有错误; stripUnknown: true 移除未定义字段; 验证中间件复用; 不仅验证 body, query 和 params 也要验证。


## 练习


<!-- Converted from: 6_Express Body 解析与验证.html -->
