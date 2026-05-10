# Express 图片处理 sharp


## 🖼️ Express 图片处理 sharp


sharp 图片处理库、resize/format/裁剪/旋转/圆角、上传时自动处理、图片水印、生成缩略图、响应式图片、WebP 转换、Stream 流式处理。


## sharp 基础


```
// ========== sharp 图片处理 ==========
// 高性能 Node.js 图片处理库 (基于 libvips)
// 比 ImageMagick 快 4-5 倍

// 安装:
// npm install sharp

const sharp = require('sharp');
const path = require('path');

// ========== 基础用法 ==========
// 打开图片 → 处理 → 输出

async function processImage() {
    // 读取 → resize → 保存
    await sharp('input.jpg')
        .resize(800, 600)     // 调整尺寸
        .toFile('output.jpg');

    // 或输出 Buffer
    const buffer = await sharp('input.jpg')
        .resize(400)
        .jpeg({ quality: 80 })
        .toBuffer();

    // 或输出到 Stream
    const readStream = sharp('input.jpg')
        .resize(200)
        .png();
}

// ========== resize 选项 ==========
await sharp('input.jpg')
    .resize({
        width: 800,
        height: 600,
        fit: 'cover',        // cover/contain/fill/inside/outside
        position: 'center',  // 裁剪位置: center/top/left/right/bottom/entropy/attention
        background: { r: 255, g: 255, b: 255 },  // 填充色 (contain 时)
        withoutEnlargement: true,  // 不放大
    })
    .toFile('output.jpg');

// fit 模式:
// cover   — 覆盖 (裁剪多余部分)
// contain — 包含 (留白)
// fill    — 拉伸 (变形)
// inside  — 缩小到完全适合
// outside — 放大到完全覆盖

// ========== 格式转换 ==========
// JPEG
await sharp('input.png')
    .jpeg({ quality: 80, progressive: true, chromaSubsampling: '4:4:4' })
    .toFile('output.jpg');

// PNG
await sharp('input.jpg')
    .png({ palette: true, quality: 90, compressionLevel: 9 })
    .toFile('output.png');

// WebP (更小体积)
await sharp('input.jpg')
    .webp({ quality: 75, effort: 6 })  // effort 0-6, 越高越小
    .toFile('output.webp');

// AVIF (最新格式)
await sharp('input.jpg')
    .avif({ quality: 70 })
    .toFile('output.avif');

// TIFF
await sharp('input.jpg')
    .tiff({ quality: 90 })
    .toFile('output.tiff');

// ========== 获取图片信息 ==========
const metadata = await sharp('input.jpg').metadata();
// {
//   format: 'jpeg',
//   width: 1920,
//   height: 1080,
//   space: 'srgb',
//   channels: 3,
//   density: 72,
//   hasAlpha: false,
//   hasProfile: true,
//   hasMetadata: true,
// }
```


## 高级操作


```
// ========== 裁剪 ==========
// 1. 按像素裁剪
await sharp('input.jpg')
    .extract({ left: 100, top: 100, width: 300, height: 300 })
    .toFile('cropped.jpg');

// 2. 按比例裁剪 (先 resize 再 extract)
// 提取中央正方形
await sharp('input.jpg')
    .resize(400, 400, { fit: 'cover', position: 'center' })
    .toFile('square.jpg');

// ========== 旋转与翻转 ==========
await sharp('input.jpg')
    .rotate(90, { background: { r: 0, g: 0, b: 0 } })  // 顺时针 90°
    .toFile('rotated.jpg');

// 自动旋转 (根据 EXIF 方向)
await sharp('input.jpg')
    .rotate()  // 无参数 = 读取 EXIF 自动旋转
    .toFile('auto-rotated.jpg');

// 翻转
await sharp('input.jpg')
    .flip()   // 垂直翻转
    .flop()   // 水平翻转
    .toFile('flipped.jpg');

// ========== 圆角与合成 ==========
// 圆角 (通过 SVG 叠加)
const roundedCorners = Buffer.from(
    ``
);

await sharp('input.jpg')
    .resize(300, 300)
    .composite([{
        input: roundedCorners,
        blend: 'dest-in',  // 仅在圆角内显示
    }])
    .png()
    .toFile('rounded.png');

// ========== 水印 ==========
// 文字水印 (通过 SVG)
const svgText = Buffer.from(`


            © My Watermark


`);

await sharp('input.jpg')
    .resize(800)
    .composite([{
        input: svgText,
        top: 0,
        left: 0,
    }])
    .toFile('watermarked.jpg');

// ========== 图片水印叠加 ==========
await sharp('input.jpg')
    .composite([{
        input: 'logo.png',
        top: 20,         // 位置
        left: 20,
        gravity: 'southeast',  // 或使用 gravity(东南)
    }])
    .toFile('with-logo.jpg');

// ========== 灰度与模糊 ==========
await sharp('input.jpg')
    .grayscale()        // 黑白
    .blur(3)            // 高斯模糊 (sigma 1-1000)
    .toFile('blur.jpg');

await sharp('input.jpg')
    .modulate({
        brightness: 1.2,  // 亮度
        saturation: 1.5,  // 饱和度
        hue: 30,          // 色相偏移
    })
    .toFile('filtered.jpg');
```


## 上传时自动处理


```
// ========== 上传 + 自动处理 ==========
// 用户上传 → sharp 处理 → 保存

const multer = require('multer');
const sharp = require('sharp');
const path = require('path');

// ========== 内存上传 + 处理 ==========
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 20 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Only images allowed'));
        }
        cb(null, true);
    },
});

app.post('/upload/image', upload.single('image'), async (req, res, next) => {
    try {
        const { buffer, originalname } = req.file;
        const name = Date.now() + '-' + Math.round(Math.random() * 1E9);

        // ========== 生成多种尺寸 ==========
        const sizes = [
            { width: 1920, suffix: '-lg' },
            { width: 800,  suffix: '-md' },
            { width: 400,  suffix: '-sm' },
            { width: 150,  suffix: '-thumb' },
        ];

        const results = await Promise.all(sizes.map(async ({ width, suffix }) => {
            const filename = `${name}${suffix}.webp`;
            await sharp(buffer)
                .resize(width, undefined, {
                    fit: 'inside',
                    withoutEnlargement: true,
                })
                .webp({ quality: 80 })
                .toFile(path.join('uploads', filename));

            return { width, filename, url: `/uploads/${filename}` };
        }));

        res.success({
            original: originalname,
            sizes: results,
        });
    } catch (err) {
        next(err);
    }
}, errorHandler);

// ========== 头像裁剪专用 ==========
app.post('/upload/avatar', upload.single('avatar'), async (req, res) => {
    const { buffer } = req.file;
    const name = Date.now() + '-avatar';

    // 400x400 居中裁剪 + WebP
    await sharp(buffer)
        .resize(400, 400, { fit: 'cover', position: 'center' })
        .webp({ quality: 85 })
        .toFile(path.join('uploads', name + '.webp'));

    // 50x50 缩略图
    await sharp(buffer)
        .resize(50, 50, { fit: 'cover', position: 'center' })
        .webp({ quality: 70 })
        .toFile(path.join('uploads', name + '-thumb.webp'));

    res.success({
        url: `/uploads/${name}.webp`,
        thumb: `/uploads/${name}-thumb.webp`,
    });
});

// ========== Stream 管道模式 (大文件) ==========
// 大文件直接用 stream, 避免内存占用

const fs = require('fs');

app.post('/upload/stream', (req, res, next) => {
    const uploadStream = multer({ storage: multer.memoryStorage() })
        .single('file')(req, res, async (err) => {
            if (err) return next(err);

            const readStream = sharp(req.file.buffer)
                .resize(1200)
                .webp({ quality: 75 });

            const writeStream = fs.createWriteStream(
                path.join('uploads', `${Date.now()}.webp`)
            );

            // 直接管道输出到响应 (动态图片处理 API)
            // readStream.pipe(res);  // 直接返回给客户端

            readStream.pipe(writeStream);
            writeStream.on('finish', () => {
                res.success({ message: 'Stream processed' });
            });
        });
});

// ========== 动态图片处理 API ==========
// /images/:filename?w=400&f=webp&q=80
app.get('/images/:filename', async (req, res) => {
    const { filename } = req.params;
    const { w = 800, f = 'webp', q = 80 } = req.query;

    const inputPath = path.join('uploads', filename);
    const width = parseInt(w);
    const quality = parseInt(q);

    res.set('Content-Type', `image/${f}`);
    res.set('Cache-Control', 'public, max-age=31536000, immutable');

    await sharp(inputPath)
        .resize(width, undefined, { withoutEnlargement: true })
        .toFormat(f, { quality })
        .pipe(res);  // 流式输出到响应
});
```


> **Note:** 💡 sharp 要点: libvips 高性能; resize 多种 fit 模式 (cover/contain/inside/outside); 格式转换 (WebP 比 JPEG 小 30%); 多尺寸生成 (lg/md/sm/thumb); 水印 composite; 圆角通过 SVG blend; 上传时自动处理; Stream 管道处理大文件; 动态图片处理 API (resize on-the-fly); 缓存控制 Cache-Control immutable。


## 练习


<!-- Converted from: 30_Express 图片处理 sharp.html -->
