# Express 自定义中间件模式


## 🔧 Express 自定义中间件模式


中间件工厂函数、可配置中间件、请求计时/日志/请求ID、条件中间件执行、中间件组合链、async/await 中间件、常见中间件模式 (认证/审计/缓存/节流)。


## 中间件工厂模式


```
// ========== 工厂模式 ==========
// 函数返回中间件, 支持配置

// ========== 请求计时器 ==========
function requestTimer(options = {}) {
    const { headerName = 'X-Response-Time', log = false } = options;

    return (req, res, next) => {
        const start = Date.now();

        // 监听响应完成事件
        res.on('finish', () => {
            const duration = Date.now() - start;

            // 自定义响应头
            res.setHeader(headerName, `${duration}ms`);

            // 慢请求日志
            if (log && duration > (options.slowThreshold || 1000)) {
                console.warn(`SLOW: ${req.method} ${req.url} - ${duration}ms`);
            }
        });

        next();
    };
}

// 使用:
app.use(requestTimer());                        // 默认
app.use(requestTimer({ headerName: 'X-Resp-Time', log: true, slowThreshold: 500 }));

// ========== 请求 ID ==========
const { v4: uuidv4 } = require('uuid');

function requestId(options = {}) {
    const {
        header = 'X-Request-Id',
        generator = uuidv4,
        setInResponse = true,
    } = options;

    return (req, res, next) => {
        // 优先用客户端传的 ID (跟踪用)
        const id = req.headers[header.toLowerCase()] || generator();

        req.id = id;
        if (setInResponse) {
            res.setHeader(header, id);
        }

        next();
    };
}

app.use(requestId());

// 后续可用 req.id 追踪请求
```


## 条件中间件


```
// ========== 条件执行 ==========
// 只在特定条件下执行中间件

// ========== 条件工厂 ==========
function unless(path, middleware) {
    return (req, res, next) => {
        const pathMatch = req.path.startsWith(path);
        if (pathMatch) {
            return next();  // 跳过
        }
        return middleware(req, res, next);
    };
}

// 使用:
app.use(unless('/health', rateLimiter));
app.use(unless('/public', authMiddleware));

// ========== 基于环境 ==========
function ifProduction(middleware) {
    return (req, res, next) => {
        if (process.env.NODE_ENV === 'production') {
            return middleware(req, res, next);
        }
        next();
    };
}

// 使用:
app.use(ifProduction(compression()));
app.use(ifProduction(helmet()));

// ========== 方法过滤 ==========
function forMethods(methods, middleware) {
    return (req, res, next) => {
        if (methods.includes(req.method)) {
            return middleware(req, res, next);
        }
        next();
    };
}

// 只对写操作执行:
app.use('/api/users', forMethods(['POST', 'PUT', 'PATCH', 'DELETE'], auditMiddleware));

// ========== 复合条件 ==========
function when(condition, middleware) {
    return (req, res, next) => {
        if (condition(req)) {
            return middleware(req, res, next);
        }
        next();
    };
}

// 使用条件函数:
const isAdmin = (req) => req.user?.role === 'admin';
const isApiRequest = (req) => req.path.startsWith('/api');

app.use(when(isAdmin, adminAuditMiddleware));
app.use(when(isApiRequest, apiMiddleware));
```


## 常见中间件模式


```
// ========== 认证中间件 ==========
function authenticate(options = {}) {
    const {
        required = true,
        roles = [],
    } = options;

    return (req, res, next) => {
        const token = req.headers.authorization?.replace('Bearer ', '');

        if (!token) {
            if (required) {
                return res.status(401).json({ message: 'Authentication required' });
            }
            return next();  // 可选认证
        }

        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            req.user = decoded;

            // 角色检查
            if (roles.length > 0 && !roles.includes(decoded.role)) {
                return res.status(403).json({ message: 'Insufficient permissions' });
            }

            next();
        } catch (err) {
            return res.status(401).json({ message: 'Invalid token' });
        }
    };
}

// 使用:
app.get('/api/users', authenticate(), listUsers);              // 必须登录
app.get('/api/users/me', authenticate({ required: false }), getMe); // 可选
app.delete('/api/users/:id', authenticate({ roles: ['admin'] }), deleteUser);

// ========== 审计日志中间件 ==========
function auditLog(eventName) {
    return async (req, res, next) => {
        const start = Date.now();

        // 拦截 res.json 记录响应
        const originalJson = res.json.bind(res);
        res.json = function(body) {
            const duration = Date.now() - start;

            // 异步记录审计日志
            AuditLog.create({
                event: eventName,
                userId: req.user?.id,
                method: req.method,
                path: req.path,
                params: req.params,
                statusCode: res.statusCode,
                duration,
                ip: req.ip,
                userAgent: req.get('User-Agent'),
                timestamp: new Date(),
            }).catch(console.error);

            return originalJson(body);
        };

        next();
    };
}

app.post('/api/users', auditLog('USER_CREATE'), createUser);
```


## 高级模式


```
// ========== 缓存中间件 ==========
const mcache = require('memory-cache');

function cache(duration = 60) {
    return (req, res, next) => {
        const key = `__cache__${req.originalUrl}`;
        const cachedBody = mcache.get(key);

        if (cachedBody) {
            return res.json(JSON.parse(cachedBody));
        }

        // 缓存原始 json 方法
        const originalJson = res.json.bind(res);
        res.json = function(body) {
            mcache.put(key, JSON.stringify(body), duration * 1000);
            return originalJson(body);
        };

        next();
    };
}

app.get('/api/products', cache(300), getProducts);  // 缓存 5 分钟

// ========== 节流 (Throttle) 中间件 ==========
function throttle(ms = 1000) {
    let lastCall = 0;

    return (req, res, next) => {
        const now = Date.now();
        if (now - lastCall < ms) {
            return res.status(429).json({ message: 'Too fast' });
        }
        lastCall = now;
        next();
    };
}

// ========== 中间件组合 ==========
// 组合多个中间件成一个:
function compose(...middlewares) {
    return (req, res, next) => {
        let index = 0;

        function run(err) {
            if (err) return next(err);
            if (index >= middlewares.length) return next();

            const middleware = middlewares[index++];
            try {
                middleware(req, res, run);
            } catch (e) {
                next(e);
            }
        }

        run();
    };
}

// 使用:
const authAndLog = compose(authMiddleware, auditLog('ACCESS'), cache(60));
app.get('/api/report', authAndLog, handler);
```


> **Note:** 💡 自定义中间件模式: 工厂函数接受配置返回中间件; 条件中间件用 unless/ifProduction/when 控制执行; 认证中间件可配置是否需要登录和角色; 审计日志拦截 res.json 记录; 缓存中间件内存缓存 GET 响应; 中间件组合链可复用中间件组合。核心: 中间件是函数, next 控制流程, 可配置可组合。


## 练习


<!-- Converted from: 11_Express 自定义中间件模式.html -->
