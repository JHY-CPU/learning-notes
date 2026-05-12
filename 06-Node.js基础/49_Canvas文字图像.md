# Canvas文字图像


## Canvas 文字图像


fillText、measureText、drawImage、getImageData、像素操作。


## 文字与图像 API


```
// ========== 文字绘制 ==========
ctx.font = 'bold 24px Arial';
ctx.fillStyle = '#333';
ctx.fillText('Hello Canvas', 50, 50);
ctx.strokeText('Hello Canvas', 50, 100);

// ========== 文字样式 ==========
ctx.textAlign = 'center';  // left/center/right
ctx.textBaseline = 'middle'; // top/middle/bottom
ctx.direction = 'ltr';     // ltr/rtl

// ========== 文字测量 ==========
const metrics = ctx.measureText('Hello');
console.log(metrics.width);   // 宽度
console.log(metrics.actualBoundingBoxAscent);
console.log(metrics.actualBoundingBoxDescent);

// ========== 图像绘制 ==========
const img = document.getElementById('myImg');
ctx.drawImage(img, 10, 10);              // 原始大小
ctx.drawImage(img, 10, 10, 100, 100);    // 缩放
ctx.drawImage(img, 0, 0, 200, 200, 10, 10, 100, 100); // 裁剪+缩放

// ========== 像素操作 ==========
const imageData = ctx.getImageData(0, 0, 100, 100);
// imageData.data = [R,G,B,A, R,G,B,A, ...]
ctx.putImageData(imageData, 200, 0);
```


## 演示：文字与图像

点击按钮查看


## 像素操作与图像滤镜

```javascript
// ========== 灰度滤镜 ==========
function grayscale(ctx, x, y, w, h) {
    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        const gray = data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114;
        data[i] = data[i+1] = data[i+2] = gray;
    }
    ctx.putImageData(imageData, x, y);
}

// ========== 反色滤镜 ==========
function invert(ctx, x, y, w, h) {
    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i+1] = 255 - data[i+1];
        data[i+2] = 255 - data[i+2];
    }
    ctx.putImageData(imageData, x, y);
}

// ========== 亮度调整 ==========
function adjustBrightness(ctx, x, y, w, h, amount) {
    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, Math.max(0, data[i] + amount));
        data[i+1] = Math.min(255, Math.max(0, data[i+1] + amount));
        data[i+2] = Math.min(255, Math.max(0, data[i+2] + amount));
    }
    ctx.putImageData(imageData, x, y);
}

// ========== 卷积核 (模糊/锐化/边缘检测) ==========
function applyKernel(ctx, x, y, w, h, kernel) {
    const imageData = ctx.getImageData(x, y, w, h);
    const data = imageData.data;
    const output = ctx.createImageData(w, h);
    const kSize = Math.sqrt(kernel.length);
    const half = Math.floor(kSize / 2);

    for (let py = 0; py < h; py++) {
        for (let px = 0; px < w; px++) {
            let r = 0, g = 0, b = 0;
            for (let ky = 0; ky < kSize; ky++) {
                for (let kx = 0; kx < kSize; kx++) {
                    const ix = Math.min(w - 1, Math.max(0, px + kx - half));
                    const iy = Math.min(h - 1, Math.max(0, py + ky - half));
                    const idx = (iy * w + ix) * 4;
                    const weight = kernel[ky * kSize + kx];
                    r += data[idx] * weight;
                    g += data[idx + 1] * weight;
                    b += data[idx + 2] * weight;
                }
            }
            const outIdx = (py * w + px) * 4;
            output.data[outIdx] = Math.min(255, Math.max(0, r));
            output.data[outIdx + 1] = Math.min(255, Math.max(0, g));
            output.data[outIdx + 2] = Math.min(255, Math.max(0, b));
            output.data[outIdx + 3] = 255;
        }
    }
    ctx.putImageData(output, x, y);
}

// 常用卷积核
const kernels = {
    blur: [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9],
    sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    edge: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
};
```

## 文字排版与水印

```javascript
// ========== 多行文字 ==========
function drawMultilineText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split('');
    let line = '';
    let currentY = y;

    for (const char of words) {
        const testLine = line + char;
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && line) {
            ctx.fillText(line, x, currentY);
            line = char;
            currentY += lineHeight;
        } else {
            line = testLine;
        }
    }
    ctx.fillText(line, x, currentY);
}

// ========== 图片水印 ==========
function addWatermark(ctx, text, canvasWidth, canvasHeight) {
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.font = '20px Arial';
    ctx.fillStyle = '#000';
    ctx.translate(canvasWidth / 2, canvasHeight / 2);
    ctx.rotate(-Math.PI / 6);

    for (let y = -canvasHeight; y < canvasHeight * 2; y += 80) {
        for (let x = -canvasWidth; x < canvasWidth * 2; x += 200) {
            ctx.fillText(text, x, y);
        }
    }
    ctx.restore();
}

// ========== Canvas 转图片 ==========
const dataURL = canvas.toDataURL('image/png');     // base64
const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
```

<!-- Converted from: 49_Canvas文字图像.html -->
