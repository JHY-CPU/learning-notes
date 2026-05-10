# Express dotenv 多环境配置


## ⚙️ Express dotenv 多环境配置


dotenv 环境变量管理、多环境文件 (.env.development/.env.production)、配置验证、配置对象封装、环境切换、敏感信息保护、12-Factor App 配置原则。


## dotenv 基础


```
// ========== dotenv 环境变量 ==========
// 通过 .env 文件管理配置
// 不上传到 git!

// 安装:
// npm install dotenv

// ========== .env 文件 ==========
// .env (基础配置, 所有环境共享)
// PORT=3000
// NODE_ENV=development
// API_PREFIX=/api/v1
//
// .env.development (开发环境)
// DB_URI=mongodb://localhost:27017/myapp
// REDIS_URL=redis://localhost:6379
// LOG_LEVEL=debug
//
// .env.production (生产环境)
// DB_URI=mongodb+srv://user:pass@cluster.mongodb.net/myapp
// REDIS_URL=redis://redis-prod:6379
// LOG_LEVEL=warn
// S3_BUCKET=myapp-prod
// CDN_URL=https://d123.cloudfront.net
//
// .env.test (测试环境)
// DB_URI=mongodb://localhost:27017/myapp-test
// LOG_LEVEL=silent

// ========== 加载配置 ==========
// app.js:

const path = require('path');
const envFile = path.resolve(process.cwd(), `.env.${process.env.NODE_ENV || 'development'}`);

require('dotenv').config({ path: '.env' });          // 加载 .env (基础)
require('dotenv').config({ path: envFile });          // 加载环境特定配置
// 后加载的会覆盖前一个的相同变量

// 也可以直接用:
// npm install dotenv-cli
// "start:dev": "dotenv -e .env -e .env.development -- node app.js"

// ========== 配置验证 ==========
const requiredVars = [
    'DB_URI',
    'JWT_SECRET',
    'NODE_ENV',
];

const missing = requiredVars.filter(v => !process.env[v]);
if (missing.length > 0) {
    console.error(`Missing required env vars: ${missing.join(', ')}`);
    process.exit(1);
}

// ========== .gitignore ==========
// .env
// .env.development
// .env.production
// .env.local
// !.env.example  (示例文件要保留)

// ========== .env.example ==========
// 提交到 git 的配置模板:
// PORT=3000
// DB_URI=mongodb://localhost:27017/myapp
// JWT_SECRET=change-me
// REDIS_URL=redis://localhost:6379
```


## 配置对象封装


```
// ========== 配置对象 ==========
// config/index.js:
// 集中管理所有配置, 提供类型转换和默认值

const config = {
    // 服务
    env: process.env.NODE_ENV || 'development',
    port: parseInt(process.env.PORT, 10) || 3000,
    apiPrefix: process.env.API_PREFIX || '/api/v1',
    isDev: () => config.env === 'development',
    isProd: () => config.env === 'production',
    isTest: () => config.env === 'test',

    // 数据库
    db: {
        uri: process.env.DB_URI || 'mongodb://localhost:27017/myapp',
        options: {
            maxPoolSize: parseInt(process.env.DB_POOL_SIZE, 10) || 10,
            minPoolSize: 2,
            serverSelectionTimeoutMS: 5000,
        },
    },

    // Redis
    redis: {
        url: process.env.REDIS_URL || 'redis://localhost:6379',
        prefix: process.env.REDIS_PREFIX || 'myapp:',
    },

    // JWT
    jwt: {
        secret: process.env.JWT_SECRET,
        accessExpiry: process.env.JWT_ACCESS_EXPIRY || '15m',
        refreshExpiry: process.env.JWT_REFRESH_EXPIRY || '7d',
        issuer: process.env.JWT_ISSUER || 'myapp',
    },

    // 文件上传
    upload: {
        maxFileSize: parseInt(process.env.UPLOAD_MAX_SIZE, 10) || 10 * 1024 * 1024,
        storage: process.env.UPLOAD_STORAGE || 'local',
        s3: {
            bucket: process.env.S3_BUCKET,
            region: process.env.AWS_REGION || 'ap-northeast-1',
        },
    },

    // 日志
    log: {
        level: process.env.LOG_LEVEL || (config?.env === 'production' ? 'warn' : 'debug'),
        format: config?.env === 'production' ? 'json' : 'pretty',
    },

    // CORS
    cors: {
        origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:5173'],
    },
};

// 启动时验证关键配置
function validateConfig() {
    const critical = [
        ['jwt.secret', config.jwt.secret],
        ['db.uri', config.db.uri],
    ];

    const missing = critical.filter(([, val]) => !val);
    if (missing.length > 0) {
        const names = missing.map(([name]) => name).join(', ');
        throw new Error(`Missing critical configuration: ${names}`);
    }
}

// 冻结配置 (防止运行时修改)
module.exports = Object.freeze(config);

// ========== 使用 ==========
// const config = require('./config');
// mongoose.connect(config.db.uri, config.db.options);
// app.listen(config.port);
```


## 环境切换与脚本


```
// ========== package.json 脚本 ==========
// {
//   "scripts": {
//     "start": "node app.js",
//     "start:dev": "NODE_ENV=development node app.js",
//     "start:prod": "NODE_ENV=production node app.js",
//     "start:staging": "NODE_ENV=staging node app.js",
//     "test": "NODE_ENV=test jest",
//     "lint": "NODE_ENV=test eslint ."
//   }
// }

// Windows 需要 cross-env:
// npm install cross-env --save-dev
// "start:prod": "cross-env NODE_ENV=production node app.js"

// ========== 环境检测 ==========
// 应用中根据环境调整行为:
if (config.isDev()) {
    // 开发: 详细日志, 不压缩, CORS 宽松
    app.use(morgan('dev'));
    app.use(cors({ origin: true, credentials: true }));
} else {
    // 生产: JSON 日志, 压缩, CORS 严格
    app.use(morgan('combined'));
    app.use(compression());
    app.use(helmet());
    app.use(cors({
        origin: config.cors.origin,
        credentials: true,
        maxAge: 86400,
    }));
}

// ========== 敏感信息 ==========
// 1. .env 文件永远不提交到 git
// 2. 生产环境用环境变量 (CI/CD 注入)
// 3. 密钥轮换 (定期更换 JWT_SECRET)
// 4. 不要打印环境变量
// 5. 配置对象不序列化到响应

// ========== 12-Factor App 配置 ==========
// 1. 配置与代码严格分离
// 2. 不把配置写死在代码里
// 3. 通过环境变量注入
// 4. 支持不修改代码切换环境
// 5. 环境特定文件: .env.development / .env.production
// 6. CI/CD 中通过 secrets 注入

// ========== Docker 环境变量 ==========
// docker-compose.yml:
// services:
//   app:
//     image: myapp
//     environment:
//       - NODE_ENV=production
//       - DB_URI=mongodb://db:27017/myapp
//       - JWT_SECRET=${JWT_SECRET}
//     env_file:
//       - .env.production
//
// 或 Dockerfile 中:
// ENV NODE_ENV=production
```


> **Note:** 💡 dotenv 要点: .env 不上传 git; 多环境文件 (.development/.production/.test); dotenv.config({ path }) 加载; 配置对象封装 类型转换+默认值; 启动时验证关键配置; cross-env 跨平台设置 NODE_ENV; 12-Factor App 配置分离; Docker env_file; .env.example 提交模板; 敏感信息不打印不序列化。


## 练习


<!-- Converted from: 35_Express dotenv 多环境配置.html -->
