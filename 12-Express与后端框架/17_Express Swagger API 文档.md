# Express Swagger API 文档


## 📖 Express Swagger API 文档


OpenAPI 3.0 规范、swagger-jsdoc + swagger-ui-express 集成、API 注解 (paths/components/schemas)、JSDoc 注释生成文档、请求/响应模型定义、Bearer Token 认证、API 分组与标签。


## OpenAPI 与 Swagger


```
// ========== OpenAPI ==========
//  REST API 的规范标准 (原名 Swagger)
//  3.0 是目前主要版本

// ========== 核心概念 ==========
// OpenAPI 文档描述:
// 1. API 信息 (title/version/description)
// 2. 服务器地址 (servers)
// 3. 路径与操作 (paths + methods)
// 4. 请求/响应模型 (components/schemas)
// 5. 安全方案 (securitySchemes)
// 6. 标签分组 (tags)

// ========== 安装 ==========
npm install swagger-jsdoc swagger-ui-express

// ========== 基础配置 ==========
// swagger.js:
const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');

const options = {
    definition: {
        openapi: '3.0.0',
        info: {
            title: 'My Express API',
            version: '1.0.0',
            description: 'REST API documentation',
            contact: {
                name: 'Developer',
                email: 'dev@example.com',
            },
        },
        servers: [
            { url: 'http://localhost:3000', description: 'Development' },
            { url: 'https://api.example.com', description: 'Production' },
        ],
        components: {
            securitySchemes: {
                bearerAuth: {
                    type: 'http',
                    scheme: 'bearer',
                    bearerFormat: 'JWT',
                },
            },
        },
        security: [{ bearerAuth: [] }],
    },
    apis: ['./src/routes/*.js', './src/models/*.js'], // 扫描路径
};

const specs = swaggerJsdoc(options);

module.exports = { specs, swaggerUi };
```


## 路由注解


```
// ========== JSDoc 注解 ==========
// 在路由文件里用 JSDoc 注释描述 API

// routes/userRoutes.js:

/**
 * @swagger
 * components:
 *   schemas:
 *     User:
 *       type: object
 *       required:
 *         - name
 *         - email
 *       properties:
 *         _id:
 *           type: string
 *           description: Auto-generated ID
 *         name:
 *           type: string
 *           description: User name
 *         email:
 *           type: string
 *           format: email
 *         role:
 *           type: string
 *           enum: [user, admin]
 *         createdAt:
 *           type: string
 *           format: date-time
 *     Error:
 *       type: object
 *       properties:
 *         success:
 *           type: boolean
 *         code:
 *           type: string
 *         message:
 *           type: string
 *         requestId:
 *           type: string
 */

/**
 * @swagger
 * /api/users:
 *   get:
 *     summary: Get user list
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           default: 1
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *       - in: query
 *         name: sort
 *         schema:
 *           type: string
 *         description: Sort field (prefix - for desc)
 *       - in: query
 *         name: role
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: User list
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/User'
 *                 pagination:
 *                   $ref: '#/components/schemas/Pagination'
 */
router.get('/', authenticate, userController.list);
```


## CRUD 完整注解


```
// ========== CRUD 注解 ==========

/**
 * @swagger
 * /api/users/{id}:
 *   get:
 *     summary: Get user by ID
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: User details
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 data:
 *                   $ref: '#/components/schemas/User'
 *       404:
 *         description: User not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
router.get('/:id', authenticate, userController.getById);

/**
 * @swagger
 * /api/users:
 *   post:
 *     summary: Create a new user
 *     tags: [Users]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - email
 *               - password
 *             properties:
 *               name:
 *                 type: string
 *               email:
 *                 type: string
 *                 format: email
 *               password:
 *                 type: string
 *                 minLength: 8
 *               role:
 *                 type: string
 *                 enum: [user, admin]
 *     responses:
 *       201:
 *         description: User created
 *       422:
 *         description: Validation error
 */
router.post('/', authenticate, authorize('admin'), validate(createUserSchema), userController.create);

/**
 * @swagger
 * /api/users/{id}:
 *   put:
 *     summary: Update user
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *               email:
 *                 type: string
 *     responses:
 *       200:
 *         description: User updated
 */
router.put('/:id', authenticate, authorize('admin'), userController.update);

/**
 * @swagger
 * /api/users/{id}:
 *   delete:
 *     summary: Delete user
 *     tags: [Users]
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *     responses:
 *       200:
 *         description: User deleted
 */
router.delete('/:id', authenticate, authorize('admin'), userController.delete);
```


## 集成与最佳实践


```
// ========== 挂载 Swagger UI ==========
// app.js:
const { specs, swaggerUi } = require('./swagger');

// Swagger 文档页面:
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs, {
    explorer: true,
    customCss: '.swagger-ui .topbar { display: none }',
    customSiteTitle: 'My API Docs',
}));

// JSON 格式文档:
app.get('/api-docs.json', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.send(specs);
});

// ========== 环境控制 ==========
// 只在非生产环境暴露:
if (process.env.NODE_ENV !== 'production') {
    app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(specs));
}

// ========== 分页 Schema ==========
// 可以在任何路由文件中定义一次:
/**
 * @swagger
 * components:
 *   schemas:
 *     Pagination:
 *       type: object
 *       properties:
 *         page:
 *           type: integer
 *         limit:
 *           type: integer
 *         total:
 *           type: integer
 *         totalPages:
 *           type: integer
 *         hasNext:
 *           type: boolean
 *         hasPrev:
 *           type: boolean
 */

// ========== 最佳实践 ==========
// 1. 用 @swagger 注释路由, 代码与文档在一起
// 2. components/schemas 定义可复用的模型
// 3. tags 分组路由 (Users/Orders/Products)
// 4. 所有 API 都需要描述 responses
// 5. securitySchemes 定义 JWT Bearer
// 6. 生产环境隐藏 API 文档
// 7. JSON 格式可被其他工具消费 (Postman)
// 8. 用 swagger-autogen 自动生成
```


> **Note:** 💡 Swagger 要点: OpenAPI 3.0 是 REST API 标准; swagger-jsdoc 从 JSDoc 注释生成文档; swagger-ui-express 渲染交互式页面; components/schemas 定义可复用模型; securitySchemes 配置 JWT; tags 分组路由; 生产环境隐藏文档; JSON 格式可互导 Postman; 代码与文档同处, 保持同步。


## 练习


<!-- Converted from: 17_Express Swagger API 文档.html -->
