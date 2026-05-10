# Express S3 与云存储


## ☁️ Express S3 与云存储


AWS S3 SDK (@aws-sdk/client-s3)、文件上传/下载/删除、预签名 URL (Presigned URL)、S3 静态网站托管、MinIO 本地兼容、CDN 加速 (CloudFront)、文件管理 API、权限控制。


## AWS S3 SDK v3


```
// ========== AWS S3 对象存储 ==========
// 安装:
// npm install @aws-sdk/client-s3 @aws-sdk/s3-request-presigner

const {
    S3Client,
    PutObjectCommand,
    GetObjectCommand,
    DeleteObjectCommand,
    ListObjectsV2Command,
    HeadObjectCommand,
} = require('@aws-sdk/client-s3');

const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

// ========== S3 客户端配置 ==========
const s3 = new S3Client({
    region: process.env.AWS_REGION || 'ap-northeast-1',
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    },
    // MinIO 本地兼容配置:
    // endpoint: 'http://localhost:9000',
    // forcePathStyle: true,
});

const BUCKET = process.env.S3_BUCKET || 'my-app-uploads';

// ========== 上传文件 ==========
async function uploadFile(buffer, filename, mimetype) {
    const key = `uploads/${Date.now()}-${filename}`;

    const cmd = new PutObjectCommand({
        Bucket: BUCKET,
        Key: key,
        Body: buffer,
        ContentType: mimetype,
        // ACL: 'public-read',  // 公开可读
        // Metadata: { uploadedBy: userId },
    });

    await s3.send(cmd);

    // 返回公开 URL
    return {
        key,
        url: `https://${BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`,
    };
}

// ========== Express 上传端点 ==========
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage() });

app.post('/api/files/upload', authenticate, upload.single('file'), async (req, res) => {
    const { buffer, originalname, mimetype } = req.file;

    const result = await uploadFile(buffer, originalname, mimetype);

    // 保存文件元数据到数据库
    const file = await File.create({
        userId: req.user.sub,
        key: result.key,
        url: result.url,
        name: originalname,
        size: req.file.size,
        mimetype,
    });

    res.created(file);
});

// ========== 下载文件 ==========
app.get('/api/files/:id/download', authenticate, async (req, res) => {
    const file = await File.findOne({ _id: req.params.id, userId: req.user.sub });
    if (!file) throw new NotFoundError('File not found');

    const cmd = new GetObjectCommand({
        Bucket: BUCKET,
        Key: file.key,
    });

    const response = await s3.send(cmd);

    // 流式返回
    res.setHeader('Content-Type', response.ContentType);
    res.setHeader('Content-Disposition', `attachment; filename="${file.name}"`);
    response.Body.pipe(res);
});

// ========== 删除文件 ==========
app.delete('/api/files/:id', authenticate, async (req, res) => {
    const file = await File.findOne({ _id: req.params.id, userId: req.user.sub });
    if (!file) throw new NotFoundError('File not found');

    await s3.send(new DeleteObjectCommand({
        Bucket: BUCKET,
        Key: file.key,
    }));

    await file.deleteOne();
    res.noContent();
});

// ========== 文件列表 ==========
app.get('/api/files', authenticate, async (req, res) => {
    const files = await File.find({ userId: req.user.sub })
        .sort({ createdAt: -1 })
        .limit(50);

    res.success(files);
});
```


## 预签名 URL


```
// ========== 预签名 URL ==========
// 无需暴露 AWS 凭证, 客户端直接上传/下载
// URL 有有效期, 过期后失效

// ========== 生成上传预签名 URL ==========
const { PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

async function getPresignedUploadUrl(key, contentType, expiresIn = 3600) {
    const cmd = new PutObjectCommand({
        Bucket: BUCKET,
        Key: key,
        ContentType: contentType,
    });

    const url = await getSignedUrl(s3, cmd, { expiresIn });  // 秒
    return url;
}

// API: 客户端请求上传 URL
app.post('/api/files/presigned-url', authenticate, async (req, res) => {
    const { filename, contentType } = req.body;

    const key = `uploads/${req.user.sub}/${Date.now()}-${filename}`;
    const uploadUrl = await getPresignedUploadUrl(key, contentType, 3600);

    res.success({
        uploadUrl,          // 客户端直接用 PUT + 此 URL 上传
        publicUrl: `https://${BUCKET}.s3.amazonaws.com/${key}`,
        key,
    });
});

// ========== 客户端使用预签名 URL ==========
// 前端:
// const { uploadUrl } = await fetch('/api/files/presigned-url', {
//     method: 'POST',
//     headers: { Authorization: `Bearer ${token}` },
//     body: JSON.stringify({ filename: 'photo.jpg', contentType: 'image/jpeg' }),
// }).then(r => r.json()).then(r => r.data);
//
// // 直接 PUT 到 S3
// await fetch(uploadUrl, {
//     method: 'PUT',
//     body: file,
//     headers: { 'Content-Type': file.type },
// });

// ========== 下载预签名 URL ==========
async function getPresignedDownloadUrl(key, expiresIn = 3600) {
    const cmd = new GetObjectCommand({
        Bucket: BUCKET,
        Key: key,
    });

    return await getSignedUrl(s3, cmd, { expiresIn });
}

app.get('/api/files/:id/download-url', authenticate, async (req, res) => {
    const file = await File.findOne({ _id: req.params.id, userId: req.user.sub });
    if (!file) throw new NotFoundError('File not found');

    const downloadUrl = await getPresignedDownloadUrl(file.key, 300);  // 5分钟

    res.success({
        downloadUrl,
        expiresIn: 300,
    });
});

// ========== 预签名优势 ==========
// 1. 服务器不经过文件内容 (减轻负载)
// 2. 客户端直连 S3 (速度快)
// 3. 无需公开 bucket
// 4. 自动过期 (安全)
// 5. 支持断点续传 (大文件)
```


## MinIO 本地开发


```
// ========== MinIO ==========
// S3 兼容的本地对象存储
// 开发环境替代 AWS S3

// ========== Docker 启动 ==========
// docker run -p 9000:9000 -p 9001:9001 \
//   -e MINIO_ROOT_USER=minioadmin \
//   -e MINIO_ROOT_PASSWORD=minioadmin \
//   quay.io/minio/minio server /data --console-address ":9001"

// ========== MinIO 客户端配置 ==========
const { S3Client } = require('@aws-sdk/client-s3');

const s3Client = new S3Client({
    region: 'us-east-1',
    endpoint: process.env.S3_ENDPOINT || 'http://localhost:9000',
    forcePathStyle: true,  // MinIO 需要!
    credentials: {
        accessKeyId: process.env.S3_ACCESS_KEY || 'minioadmin',
        secretAccessKey: process.env.S3_SECRET_KEY || 'minioadmin',
    },
});

// ========== Bucket 初始化 ==========
const { CreateBucketCommand, HeadBucketCommand } = require('@aws-sdk/client-s3');

async function ensureBucket(bucket) {
    try {
        await s3Client.send(new HeadBucketCommand({ Bucket: bucket }));
    } catch (err) {
        if (err.name === 'NotFound') {
            await s3Client.send(new CreateBucketCommand({
                Bucket: bucket,
                // ACL: 'public-read',
            }));
            console.log(`Bucket ${bucket} created`);
        }
    }
}

// 启动时初始化
ensureBucket(process.env.S3_BUCKET || 'my-app');

// ========== 环境切换 ==========
// .env.development:
// S3_ENDPOINT=http://localhost:9000
// S3_ACCESS_KEY=minioadmin
// S3_SECRET_KEY=minioadmin
// S3_BUCKET=my-app-dev
// S3_FORCE_PATH_STYLE=true
//
// .env.production:
// S3_REGION=ap-northeast-1
// S3_ACCESS_KEY=AKIA...
// S3_SECRET_KEY=...
// S3_BUCKET=my-app-prod
```


## CDN 与文件管理


```
// ========== CloudFront CDN ==========
// S3 + CloudFront = 全球加速 + HTTPS

// CloudFront 域名:
// https://d123.cloudfront.net/uploads/file.jpg

// ========== 文件存储路径策略 ==========
// 按用户/日期组织:
// uploads/{userId}/{year}/{month}/{day}/{filename}

function buildKey(userId, filename) {
    const now = new Date();
    const datePath = `${now.getFullYear()}/${String(now.getMonth()+1).padStart(2,'0')}`;
    const uniqueId = crypto.randomUUID().slice(0, 8);
    return `uploads/${userId}/${datePath}/${uniqueId}-${filename}`;
}

// ========== 文件类型与大小限制 ==========
const ALLOWED_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'application/pdf': '.pdf',
    'application/zip': '.zip',
    'text/csv': '.csv',
    'application/json': '.json',
};

const MAX_FILE_SIZE = 50 * 1024 * 1024;  // 50MB

function validateFileType(mimetype) {
    return !!ALLOWED_TYPES[mimetype];
}

// ========== 批量清理 ==========
// 清理未关联的文件 (定时任务)
async function cleanupOrphanedFiles() {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);

    const orphans = await File.find({
        createdAt: { $lt: oneDayAgo },
        status: 'temporary',
    });

    for (const file of orphans) {
        await s3.send(new DeleteObjectCommand({
            Bucket: BUCKET,
            Key: file.key,
        }));
        await file.deleteOne();
    }
}

// 每小时执行
// setInterval(cleanupOrphanedFiles, 60 * 60 * 1000);

// ========== 文件夹管理 (S3 没有真正文件夹) ==========
// S3 是扁平键值存储
// "folder/" 只是前缀, 通过 / 分隔模拟目录

// 列出某用户所有文件:
async function listUserFiles(userId) {
    const prefix = `uploads/${userId}/`;

    const cmd = new ListObjectsV2Command({
        Bucket: BUCKET,
        Prefix: prefix,
        // Delimiter: '/',  // 按目录分组
        MaxKeys: 100,
    });

    const response = await s3.send(cmd);

    return (response.Contents || []).map(obj => ({
        key: obj.Key,
        size: obj.Size,
        lastModified: obj.LastModified,
        url: `https://cdn.example.com/${obj.Key}`,
    }));
}
```


> **Note:** 💡 S3 要点: @aws-sdk/client-s3 v3 模块化; PutObjectCommand 上传; GetObjectCommand 流式下载; 预签名 URL 客户端直传; MinIO 本地开发 (forcePathStyle); 路径策略 按用户/日期组织; CloudFront CDN 加速; 文件类型白名单; 定时清理过期文件; S3 扁平存储 / 前缀模拟目录。


## 练习


<!-- Converted from: 31_Express S3 与云存储.html -->
