# Express 统一响应与错误格式


## 📋 Express 统一响应与错误格式


统一 API 响应规范 (成功/错误/分页格式)、响应辅助方法、错误码体系、生产错误屏蔽、多语言错误消息、验证错误格式化、API 请求 ID 追踪。


## 统一响应规范


```
// ========== 响应规范 ==========
// 所有 API 遵循统一 JSON 响应结构

// ========== 成功响应 ==========
// 单个资源:
{
    "success": true,
    "data": { "id": 1, "name": "Alice" },
    "message": "Success",
    "requestId": "req_abc123"
}

// 列表:
{
    "success": true,
    "data": [ ... ],
    "pagination": {
        "page": 1,
        "limit": 20,
        "total": 100,
        "totalPages": 5,
        "hasNext": true
    },
    "requestId": "req_abc123"
}

// ========== 错误响应 ==========
// 验证错误:
{
    "success": false,
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "errors": [
        { "field": "email", "message": "Invalid email format", "code": "invalid_string" }
    ],
    "requestId": "req_abc123"
}

// 业务错误:
{
    "success": false,
    "code": "INSUFFICIENT_INVENTORY",
    "message": "Insufficient inventory for product iPhone 15",
    "details": { "productId": "p123", "available": 0, "requested": 2 },
    "requestId": "req_abc123"
}

// 服务器错误:
{
    "success": false,
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "requestId": "req_abc123"
    // 生产环境不暴露 stack!
}
```


## 响应辅助方法


```
// ========== 响应增强 ==========
// 为 res 对象添加辅助方法

// middleware/response.js:
function responseEnhancer(req, res, next) {
    // 成功 (单个)
    res.success = function(data, message = 'Success', status = 200) {
        return res.status(status).json({
            success: true,
            data,
            message,
            requestId: req.id,
        });
    };

    // 成功 (分页)
    res.paginated = function(data, pagination, message = 'Success') {
        return res.status(200).json({
            success: true,
            data,
            pagination,
            message,
            requestId: req.id,
        });
    };

    // 创建成功
    res.created = function(data, message = 'Created') {
        return res.status(201).json({
            success: true,
            data,
            message,
            requestId: req.id,
        });
    };

    // 无内容
    res.noContent = function() {
        return res.status(204).end();
    };

    // 错误
    res.fail = function(message, code = 'BAD_REQUEST', status = 400, details = null) {
        const body = {
            success: false,
            code,
            message,
            requestId: req.id,
        };
        if (details) body.details = details;
        return res.status(status).json(body);
    };

    next();
}

// app.js:
app.use(responseEnhancer);

// ========== 控制器中使用 ==========
app.get('/users/:id', asyncHandler(async (req, res) => {
    const user = await User.findById(req.params.id).lean();
    if (!user) return res.fail('User not found', 'NOT_FOUND', 404);

    res.success(user);
}));

app.post('/users', asyncHandler(async (req, res) => {
    const user = await User.create(req.body);
    res.created(user);
}));
```


## 错误码体系


```
// ========== 错误码定义 ==========
// constants/errorCodes.js:
const ErrorCodes = {
    // 通用 (4xx)
    BAD_REQUEST: { code: 'BAD_REQUEST', status: 400, message: 'Bad request' },
    UNAUTHORIZED: { code: 'UNAUTHORIZED', status: 401, message: 'Authentication required' },
    FORBIDDEN: { code: 'FORBIDDEN', status: 403, message: 'Insufficient permissions' },
    NOT_FOUND: { code: 'NOT_FOUND', status: 404, message: 'Resource not found' },
    METHOD_NOT_ALLOWED: { code: 'METHOD_NOT_ALLOWED', status: 405, message: 'Method not allowed' },
    CONFLICT: { code: 'CONFLICT', status: 409, message: 'Resource conflict' },
    VALIDATION_ERROR: { code: 'VALIDATION_ERROR', status: 422, message: 'Validation failed' },
    TOO_MANY_REQUESTS: { code: 'TOO_MANY_REQUESTS', status: 429, message: 'Rate limit exceeded' },

    // 业务
    INSUFFICIENT_INVENTORY: { code: 'INSUFFICIENT_INVENTORY', status: 409, message: 'Insufficient inventory' },
    DUPLICATE_EMAIL: { code: 'DUPLICATE_EMAIL', status: 409, message: 'Email already exists' },
    INVALID_CREDENTIALS: { code: 'INVALID_CREDENTIALS', status: 401, message: 'Invalid credentials' },
    TOKEN_EXPIRED: { code: 'TOKEN_EXPIRED', status: 401, message: 'Token has expired' },
    ORDER_NOT_PAYABLE: { code: 'ORDER_NOT_PAYABLE', status: 400, message: 'Order cannot be paid' },

    // 服务端 (5xx)
    INTERNAL_ERROR: { code: 'INTERNAL_ERROR', status: 500, message: 'Internal server error' },
    SERVICE_UNAVAILABLE: { code: 'SERVICE_UNAVAILABLE', status: 503, message: 'Service temporarily unavailable' },
};

// ========== 错误响应构建 ==========
// utils/errorResponse.js:
function buildErrorResponse(errorCode, override = {}, req) {
    const { code, status, message } = ErrorCodes[errorCode] || ErrorCodes.INTERNAL_ERROR;

    return {
        success: false,
        code: code,
        message: override.message || message,
        status: override.status || status,
        errors: override.errors || undefined,
        details: override.details || undefined,
        requestId: req?.id,
    };
}

// 在错误处理中间件中使用:
app.use((err, req, res, next) => {
    const errorResponse = buildErrorResponse(err.errorCode || 'INTERNAL_ERROR', {
        message: err.isOperational ? err.message : undefined,
        errors: err.errors,
        details: err.details,
    }, req);

    res.status(errorResponse.status).json(errorResponse);
});
```


## 生产错误屏蔽与日志


```
// ========== 生产错误处理 ==========
// middleware/errorHandler.js (生产版):

function errorHandler(err, req, res, next) {
    // 1. 记录完整错误日志
    logger.error({
        err,
        requestId: req.id,
        method: req.method,
        url: req.originalUrl,
        body: req.body,
        user: req.user?.id,
        ip: req.ip,
    });

    // 2. 确定状态码和错误码
    let statusCode = err.statusCode || 500;
    let code = err.code || 'INTERNAL_ERROR';
    let message = err.message || 'Internal server error';

    // 3. 转换已知错误类型
    if (err.name === 'ValidationError') {           // Mongoose
        statusCode = 422;
        code = 'VALIDATION_ERROR';
        const errors = Object.values(err.errors).map(e => ({
            field: e.path,
            message: e.message,
        }));
        return res.status(422).json({
            success: false, code, message: 'Validation failed',
            errors, requestId: req.id,
        });
    }

    if (err.code === 11000) {                       // MongoDB duplicate
        statusCode = 409;
        code = 'DUPLICATE_KEY';
        const field = Object.keys(err.keyValue)[0];
        message = `${field} already exists`;
    }

    if (err.name === 'JsonWebTokenError') {
        statusCode = 401;
        code = 'INVALID_TOKEN';
        message = 'Invalid token';
    }

    if (err.name === 'CastError') {                 // Invalid ObjectId
        statusCode = 400;
        code = 'INVALID_ID';
        message = 'Invalid ID format';
    }

    // 4. 生产环境隐藏内部错误
    const response = {
        success: false,
        code,
        message: process.env.NODE_ENV === 'production' && statusCode === 500
            ? 'Internal server error'
            : message,
        requestId: req.id,
    };

    // 开发环境附加调试信息
    if (process.env.NODE_ENV !== 'production') {
        response.stack = err.stack;
        response.details = err.details;
    }

    res.status(statusCode).json(response);
}

module.exports = errorHandler;
```


> **Note:** 💡 统一响应要点: 所有 API 返回一致结构 (success/data/pagination/error); 响应辅助方法 (res.success/fail/paginated/created) 减少重复; 错误码体系让前端可程序化处理; 生产环境隐藏内部错误详情; 请求 ID 追踪错误链路; 转换已知错误类型 (Mongoose/MongoDB/JWT); 完整错误日志含上下文。


## 练习


<!-- Converted from: 16_Express 统一响应与错误格式.html -->
