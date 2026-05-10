# Express 文件上传 multer


## 📤 Express 文件上传 multer


multer 中间件配置、diskStorage vs memoryStorage、单文件/多文件上传、文件类型与大小验证、错误处理、自定义文件名、前端上传配合。


## multer 基础


```
// ========== multer 文件上传 ==========
// HTTP 文件上传通过 multipart/form-data
// multer 是 Express 最流行的上传中间件

// 安装:
// npm install multer

const multer = require('multer');
const path = require('path');

// ========== 磁盘存储 (diskStorage) ==========
const storage = multer.diskStorage({
    // 保存目录
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    // 文件名 (防止重名)
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    },
});

const upload = multer({
    storage,
    limits: {
        fileSize: 5 * 1024 * 1024,  // 5MB
    },
    fileFilter: (req, file, cb) => {
        const allowed = /jpeg|jpg|png|gif|webp/;
        const extOk = allowed.test(path.extname(file.originalname).toLowerCase());
        const mimeOk = allowed.test(file.mimetype);

        if (extOk && mimeOk) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    },
});

// ========== 单文件上传 ==========
// 表单字段名: avatar
app.post('/upload/avatar', upload.single('avatar'), (req, res) => {
    // req.file 包含上传的文件信息
    res.success({
        filename: req.file.filename,
        url: `/uploads/${req.file.filename}`,
        size: req.file.size,
    });
});

// ========== 多文件上传 ==========
// 多个相同字段:
app.post('/upload/photos', upload.array('photos', 10), (req, res) => {
    // req.files 是文件数组
    const files = req.files.map(f => ({
        url: `/uploads/${f.filename}`,
        size: f.size,
    }));
    res.success({ files, count: files.length });
});

// 多个不同字段:
app.post('/upload/documents', upload.fields([
    { name: 'avatar', maxCount: 1 },
    { name: 'gallery', maxCount: 8 },
    { name: 'attachment', maxCount: 3 },
]), (req, res) => {
    // req.files['avatar'], req.files['gallery'], req.files['attachment']
    res.success({
        avatar: req.files['avatar']?.[0]?.filename,
        gallery: req.files['gallery']?.map(f => f.filename),
    });
});

// ========== multer 返回的文件信息 ==========
// req.file = {
//   fieldname: 'avatar',
//   originalname: 'profile.jpg',
//   encoding: '7bit',
//   mimetype: 'image/jpeg',
//   destination: 'uploads/',
//   filename: '1712345678901-123456789.jpg',
//   path: 'uploads/1712345678901-123456789.jpg',
//   size: 102400
// }
```


## 内存存储与 Buffer


```
// ========== 内存存储 (memoryStorage) ==========
// 文件不保存到磁盘, 直接操作 Buffer
// 适用于: 云存储上传 (S3)、图片处理、校验

const memoryUpload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 },
});

// ========== 上传到 S3 (内存流式) ==========
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');

const s3 = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY,
        secretAccessKey: process.env.AWS_SECRET_KEY,
    },
});

app.post('/upload/s3', memoryUpload.single('file'), async (req, res) => {
    const ext = path.extname(req.file.originalname);
    const key = `uploads/${Date.now()}${ext}`;

    await s3.send(new PutObjectCommand({
        Bucket: process.env.S3_BUCKET,
        Key: key,
        Body: req.file.buffer,
        ContentType: req.file.mimetype,
        ACL: 'public-read',
    }));

    const url = `https://${process.env.S3_BUCKET}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`;

    res.success({
        url,
        key,
        size: req.file.size,
    });
});

// ========== Base64 转换 ==========
app.post('/upload/base64', memoryUpload.single('file'), (req, res) => {
    const base64 = req.file.buffer.toString('base64');
    const dataUrl = `data:${req.file.mimetype};base64,${base64}`;

    res.success({
        dataUrl,
        size: req.file.size,
    });
});

// ========== 文件信息校验 (上传前检查) ==========
function validateFile(req, file, cb) {
    // 检查文件签名 (Magic Bytes)
    // 更安全的文件类型检测

    const allowedMimes = [
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'application/pdf',
        'application/zip',
    ];

    if (!allowedMimes.includes(file.mimetype)) {
        return cb(new Error(`File type ${file.mimetype} not allowed`));
    }

    // 自定义: 通过 buffer 检查魔数
    // (memoryStorage 配合 fileFilter 可提前读 buffer)
    cb(null, true);
}
```


## 错误处理


```
// ========== multer 错误处理 ==========
// multer 错误需要特殊处理 (不是普通 Express 错误)

// ========== MulterError 类 ==========
// const MulterError = require('multer').MulterError;

// 错误类型:
// LIMIT_FILE_SIZE     — 文件太大
// LIMIT_FILE_COUNT    — 文件数量超限
// LIMIT_UNEXPECTED_FILE — 字段名不匹配
// LIMIT_FIELD_KEY     — 字段 key 太长
// LIMIT_FIELD_VALUE   — 字段值太长
// LIMIT_FIELD_COUNT   — 字段太多
// LIMIT_PART_COUNT    — 表单部分太多

// ========== 错误处理中间件 ==========
app.post('/upload/avatar', (req, res, next) => {
    upload.single('avatar')(req, res, (err) => {
        if (err) {
            if (err instanceof multer.MulterError) {
                // multer 内部错误
                switch (err.code) {
                    case 'LIMIT_FILE_SIZE':
                        return res.status(413).fail({
                            message: 'File too large (max 5MB)',
                            code: 'FILE_TOO_LARGE',
                        });
                    case 'LIMIT_FILE_COUNT':
                        return res.status(400).fail({
                            message: 'Too many files',
                            code: 'TOO_MANY_FILES',
                        });
                    case 'LIMIT_UNEXPECTED_FILE':
                        return res.status(400).fail({
                            message: `Unexpected field: ${err.field}`,
                            code: 'UNEXPECTED_FIELD',
                        });
                    default:
                        return res.status(400).fail({
                            message: err.message,
                            code: 'UPLOAD_ERROR',
                        });
                }
            }

            // 自定义错误 (fileFilter 抛出的)
            return res.status(400).fail({
                message: err.message,
                code: 'VALIDATION_ERROR',
            });
        }

        // 上传成功
        res.success({ url: `/uploads/${req.file.filename}` });
    });
});

// ========== 全局上传错误处理 ==========
// 也可以封装为中间件:
function uploadHandler(uploadMiddleware) {
    return (req, res, next) => {
        uploadMiddleware(req, res, (err) => {
            if (err) {
                if (err instanceof multer.MulterError) {
                    return res.status(413).fail({ message: `Upload error: ${err.message}` });
                }
                return res.status(400).fail({ message: err.message });
            }
            next();
        });
    };
}

// 使用:
app.post('/upload/avatar', uploadHandler(upload.single('avatar')), (req, res) => {
    res.success({ url: `/uploads/${req.file.filename}` });
});
```


## 前端配合


```
// ========== 前端文件上传 ==========

// ========== 1. HTML Form ==========
//
//
//     Upload
//

// ========== 2. Fetch API (FormData) ==========
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('avatar', file);

    const res = await fetch('/upload/avatar', {
        method: 'POST',
        body: formData,
        // 不要设置 Content-Type!
        // 浏览器会自动设置 multipart/form-data + boundary
    });

    return res.json();
}

// ========== 3. 拖拽上传 ==========
//
//   Drag & drop files here
//

document.getElementById('dropzone').addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    for (const file of files) {
        uploadFile(file);
    }
});

document.getElementById('dropzone').addEventListener('dragover', (e) => {
    e.preventDefault();
    e.currentTarget.style.borderColor = '#3498db';
});

// ========== 4. 上传进度 ==========
function uploadWithProgress(file, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', file);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                onProgress(percent);
            }
        };

        xhr.onload = () => {
            if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
            else reject(new Error('Upload failed'));
        };

        xhr.onerror = () => reject(new Error('Network error'));

        xhr.open('POST', '/upload/file');
        xhr.send(formData);
    });
}

// 使用:
// uploadWithProgress(file, (pct) => {
//     document.getElementById('progress').style.width = pct + '%';
// });

// ========== 5. 多文件预览 ==========
// document.getElementById('file-input').addEventListener('change', (e) => {
//     const preview = document.getElementById('preview');
//     preview.innerHTML = '';
//
//     for (const file of e.target.files) {
//         if (file.type.startsWith('image/')) {
//             const img = document.createElement('img');
//             img.src = URL.createObjectURL(file);
//             img.style.maxWidth = '200px';
//             img.style.margin = '5px';
//             preview.appendChild(img);
//         }
//     }
// });
```


> **Note:** 💡 multer 要点: diskStorage 保存到磁盘; memoryStorage 操作 Buffer (S3/处理); single/array/fields 对应不同上传场景; limits.fileSize 限制大小; fileFilter 过滤类型; MulterError 需单独捕获; 前端用 FormData + fetch 或 XHR 进度; 拖拽上传 (drop/dragover); 图片预览 URL.createObjectURL。


## 练习


<!-- Converted from: 29_Express 文件上传 multer.html -->
