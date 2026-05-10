# Express 流式处理与断点续传


## 📦 Express 流式处理与断点续传


Node.js Stream (Readable/Writable/Transform)、流式上传/下载、大文件分片上传 (Multipart Upload)、断点续传 (Range Requests)、流式 CSV/JSON 处理、压缩流 (zlib)、EventSource 推送。


## Node.js Stream 基础


```
// ========== Node.js Stream ==========
// 流: 按块处理数据, 不一次性加载到内存
// 4 种流类型:
// Readable  — 可读 (读文件/请求)
// Writable  — 可写 (写文件/响应)
// Transform — 转换 (压缩/加密)
// Duplex   — 双工 (TCP)

const fs = require('fs');
const zlib = require('zlib');
const { Transform, pipeline } = require('stream');
const { promisify } = require('util');
const pipe = promisify(pipeline);

// ========== 流式读取大文件 ==========
// 传统: fs.readFile → 整个文件在内存
// 流式: fs.createReadStream → 逐块处理

// 流式响应文件下载:
app.get('/download/:file', (req, res) => {
    const filePath = path.join(__dirname, 'files', req.params.file);

    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${req.params.file}"`);

    const readStream = fs.createReadStream(filePath);

    readStream.on('error', (err) => {
        if (err.code === 'ENOENT') {
            res.status(404).json({ error: 'File not found' });
        } else {
            res.status(500).json({ error: 'Stream error' });
        }
    });

    readStream.pipe(res);
});

// ========== 流式上传 ==========
// 不用 multer, 直接接收流
app.post('/upload/stream', (req, res) => {
    const filename = req.headers['x-filename'] || `stream-${Date.now()}`;
    const writeStream = fs.createWriteStream(path.join('uploads', filename));

    let size = 0;

    req.on('data', (chunk) => {
        size += chunk.length;
        // 限流: 超过 100MB 拒绝
        if (size > 100 * 1024 * 1024) {
            req.destroy(new Error('File too large'));
            writeStream.destroy();
            res.status(413).json({ error: 'File too large' });
        }
    });

    req.pipe(writeStream);

    writeStream.on('finish', () => {
        res.json({ message: 'Upload complete', size, filename });
    });

    writeStream.on('error', (err) => {
        res.status(500).json({ error: 'Write failed' });
    });
});

// ========== Transform 流 ==========
// 处理过程中转换数据

class UppercaseTransform extends Transform {
    _transform(chunk, encoding, callback) {
        this.push(chunk.toString().toUpperCase());
        callback();
    }
}

// 流式管道: 读取 → 转换(压缩) → 写入
app.get('/download/compressed', async (req, res) => {
    const filePath = path.join('uploads', 'large-file.txt');

    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Content-Encoding', 'gzip');

    try {
        await pipe(
            fs.createReadStream(filePath),
            zlib.createGzip(),         // 压缩
            new UppercaseTransform(),  // 转大写
            res                        // 输出到响应
        );
    } catch (err) {
        res.status(500).end('Stream error');
    }
});
```


## 分片上传 (Multipart Upload)


```
// ========== S3 分片上传 ==========
// 大文件分成多个 part 分别上传
// 最终合并为完整文件

const {
    CreateMultipartUploadCommand,
    UploadPartCommand,
    CompleteMultipartUploadCommand,
    AbortMultipartUploadCommand,
} = require('@aws-sdk/client-s3');

// ========== 1. 初始化分片上传 ==========
app.post('/api/files/multipart/init', authenticate, async (req, res) => {
    const { filename, contentType, totalParts } = req.body;
    const key = `uploads/${req.user.sub}/${Date.now()}-${filename}`;

    const cmd = new CreateMultipartUploadCommand({
        Bucket: BUCKET,
        Key: key,
        ContentType: contentType,
    });

    const { UploadId } = await s3.send(cmd);

    // 保存上传状态到数据库
    await MultipartUpload.create({
        userId: req.user.sub,
        uploadId: UploadId,
        key,
        filename,
        totalParts,
        status: 'initiated',
    });

    res.success({ uploadId: UploadId, key });
});

// ========== 2. 上传分片 ==========
app.post('/api/files/multipart/part', authenticate, upload.single('part'), async (req, res) => {
    const { uploadId, partNumber } = req.body;

    const upload = await MultipartUpload.findOne({ uploadId });
    if (!upload) throw new NotFoundError('Upload not found');

    const cmd = new UploadPartCommand({
        Bucket: BUCKET,
        Key: upload.key,
        UploadId: uploadId,
        PartNumber: parseInt(partNumber),
        Body: req.file.buffer,
    });

    const { ETag } = await s3.send(cmd);

    // 保存 part 信息
    await Part.create({
        uploadId,
        partNumber: parseInt(partNumber),
        etag: ETag,
    });

    res.success({ partNumber, etag: ETag });
});

// ========== 3. 完成合并 ==========
app.post('/api/files/multipart/complete', authenticate, async (req, res) => {
    const { uploadId } = req.body;

    const parts = await Part.find({ uploadId }).sort({ partNumber: 1 });
    const upload = await MultipartUpload.findOne({ uploadId });

    const cmd = new CompleteMultipartUploadCommand({
        Bucket: BUCKET,
        Key: upload.key,
        UploadId: uploadId,
        MultipartUpload: {
            Parts: parts.map(p => ({
                PartNumber: p.partNumber,
                ETag: p.etag,
            })),
        },
    });

    await s3.send(cmd);

    upload.status = 'completed';
    await upload.save();

    // 清理 part 记录
    await Part.deleteMany({ uploadId });

    res.success({
        url: `https://${BUCKET}.s3.amazonaws.com/${upload.key}`,
        key: upload.key,
    });
});

// ========== 4. 取消上传 ==========
app.post('/api/files/multipart/abort', authenticate, async (req, res) => {
    const { uploadId } = req.body;
    const upload = await MultipartUpload.findOne({ uploadId });

    const cmd = new AbortMultipartUploadCommand({
        Bucket: BUCKET,
        Key: upload.key,
        UploadId: uploadId,
    });

    await s3.send(cmd);
    upload.status = 'aborted';
    await upload.save();

    res.success({ message: 'Upload aborted' });
});

// ========== 前端分片上传 ==========
// 将文件切成 5MB 一片上传:
//
// const CHUNK_SIZE = 5 * 1024 * 1024;
//
// async function uploadLargeFile(file) {
//     const totalParts = Math.ceil(file.size / CHUNK_SIZE);
//
//     // 1. 初始化
//     const { uploadId } = await fetch('/api/files/multipart/init', {
//         method: 'POST',
//         body: JSON.stringify({ filename: file.name, totalParts }),
//         headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
//     }).then(r => r.json()).then(r => r.data);
//
//     // 2. 逐片上传
//     for (let i = 0; i < totalParts; i++) {
//         const start = i * CHUNK_SIZE;
//         const chunk = file.slice(start, start + CHUNK_SIZE);
//         const formData = new FormData();
//         formData.append('part', chunk);
//         formData.append('uploadId', uploadId);
//         formData.append('partNumber', i + 1);
//         await fetch('/api/files/multipart/part', {
//             method: 'POST',
//             body: formData,
//             headers: { Authorization: `Bearer ${token}` },
//         });
//     }
//
//     // 3. 完成
//     await fetch('/api/files/multipart/complete', {
//         method: 'POST',
//         body: JSON.stringify({ uploadId }),
//         headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
//     });
// }
```


## Range 请求与断点续传


```
// ========== HTTP Range 请求 ==========
// 支持视频/音频拖拽、断点续传、部分下载

// 客户端请求头:
// Range: bytes=0-100    (前 101 字节)
// Range: bytes=500-     (500 字节到末尾)

// 服务端响应:
// 206 Partial Content
// Content-Range: bytes 0-100/1000

// ========== 流式视频播放 (Range 支持) ==========
app.get('/video/:filename', async (req, res) => {
    const filePath = path.join('uploads', req.params.filename);
    const stat = await fs.promises.stat(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;

    if (range) {
        // 解析 Range 头
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
        const chunkSize = end - start + 1;

        res.writeHead(206, {
            'Content-Range': `bytes ${start}-${end}/${fileSize}`,
            'Accept-Ranges': 'bytes',
            'Content-Length': chunkSize,
            'Content-Type': 'video/mp4',
        });

        fs.createReadStream(filePath, { start, end }).pipe(res);
    } else {
        // 首次请求, 返回完整文件信息
        res.writeHead(200, {
            'Content-Length': fileSize,
            'Content-Type': 'video/mp4',
            'Accept-Ranges': 'bytes',
        });

        fs.createReadStream(filePath).pipe(res);
    }
});

// ========== 下载续传 ==========
app.get('/download/:filename', async (req, res) => {
    const filePath = path.join('files', req.params.filename);
    const stat = await fs.promises.stat(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;

    res.setHeader('Content-Disposition', `attachment; filename="${req.params.filename}"`);

    if (range) {
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10);
        const end = Math.min(parts[1] ? parseInt(parts[1], 10) : fileSize - 1, fileSize - 1);

        res.status(206);
        res.setHeader('Content-Range', `bytes ${start}-${end}/${fileSize}`);
        res.setHeader('Content-Length', end - start + 1);

        fs.createReadStream(filePath, { start, end }).pipe(res);
    } else {
        res.setHeader('Content-Length', fileSize);
        res.setHeader('Accept-Ranges', 'bytes');

        fs.createReadStream(filePath).pipe(res);
    }
});

// ========== 进度事件 ==========
// 客户端通过 XHR 监听进度:
// const xhr = new XMLHttpRequest();
// xhr.responseType = 'blob';
//
// xhr.onprogress = (e) => {
//     if (e.lengthComputable) {
//         console.log(`${Math.round((e.loaded / e.total) * 100)}%`);
//     }
// };
//
// xhr.open('GET', '/download/bigfile.zip');
// xhr.send();
```


## 流式 CSV/JSON 处理


```
// ========== 流式 CSV 导出 ==========
// 逐行生成, 不占用内存

const { Transform } = require('stream');

app.get('/export/users/csv', authenticate, async (req, res) => {
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="users.csv"');

    // 写入 CSV 头
    res.write('ID,Name,Email,CreatedAt\n');

    // 流式查询 (Mongoose cursor)
    const cursor = User.find().cursor();

    cursor.on('data', (user) => {
        const line = `${user._id},"${user.name}","${user.email}",${user.createdAt}\n`;
        res.write(line);
    });

    cursor.on('end', () => res.end());
    cursor.on('error', (err) => {
        res.status(500).end('Error generating CSV');
    });
});

// ========== 流式 JSON 解析 ==========
// 使用 JSONStream 解析大 JSON 文件
// npm install JSONStream

const JSONStream = require('JSONStream');

app.post('/import/data', (req, res) => {
    let count = 0;

    // 流式解析 JSON 数组
    req
        .pipe(JSONStream.parse('*'))  // 逐个解析数组元素
        .on('data', async (item) => {
            // 逐个处理, 不阻塞
            await processItem(item);
            count++;
        })
        .on('end', () => {
            res.json({ imported: count });
        })
        .on('error', (err) => {
            res.status(400).json({ error: 'Invalid JSON' });
        });
});

// ========== 流式 CSV 解析 ==========
// npm install csv-parse
const csv = require('csv-parse');

app.post('/import/csv', (req, res) => {
    const results = [];
    let count = 0;

    req
        .pipe(csv.parse({
            columns: true,      // 第一行为列名
            skip_empty_lines: true,
            trim: true,
        }))
        .on('data', (row) => {
            results.push(row);
            count++;
            // 批量插入, 每 1000 条 flush
            if (results.length >= 1000) {
                // await bulkInsert(results);
                results.length = 0;
            }
        })
        .on('end', async () => {
            if (results.length > 0) {
                // await bulkInsert(results);
            }
            res.json({ imported: count });
        });
});
```


> **Note:** 💡 流式处理要点: Stream 4 类型 (Readable/Writable/Transform/Duplex); pipeline 自动清理; pipe 方法链式组合; S3 分片上传 (CreateMultipartUpload/UploadPart/CompleteMultipartUpload); HTTP Range 206 Partial Content 视频播放; 流式 CSV 导出 (cursor); 流式 JSON/CSV 解析 (JSONStream/csv-parse); zlib 压缩流; 大文件分片前端 File.slice。


## 练习


<!-- Converted from: 32_Express 流式处理与大文件.html -->
