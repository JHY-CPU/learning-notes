# Canvas基础


## Canvas 基础


getContext、fillRect/strokeRect、颜色渐变、路径(path)、样式。


## Canvas 2D API


```
// ========== 获取上下文 ==========
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
// 也支持: 'webgl', 'webgl2', 'bitmaprenderer'

// ========== 矩形 ==========
ctx.fillStyle = 'red';
ctx.fillRect(10, 10, 100, 50);   // 填充矩形
ctx.strokeRect(10, 70, 100, 50); // 边框矩形
ctx.clearRect(15, 15, 90, 40);   // 清除区域

// ========== 路径 ==========
ctx.beginPath();
ctx.moveTo(50, 50);     // 起点
ctx.lineTo(150, 50);    // 直线
ctx.arc(100, 100, 50, 0, Math.PI * 2); // 圆
ctx.closePath();
ctx.fill();    // 填充
ctx.stroke();  // 描边

// ========== 样式 ==========
ctx.fillStyle = '#3498db';   // 填充色
ctx.strokeStyle = '#2c3e50'; // 描边色
ctx.lineWidth = 2;           // 线宽
ctx.globalAlpha = 0.5;       // 全局透明度
```


## 演示：Canvas

点击按钮查看


## 渐变与复杂路径

```javascript
// ========== 线性渐变 ==========
const gradient = ctx.createLinearGradient(0, 0, 200, 0);
gradient.addColorStop(0, '#ff0000');
gradient.addColorStop(0.5, '#00ff00');
gradient.addColorStop(1, '#0000ff');
ctx.fillStyle = gradient;
ctx.fillRect(10, 10, 200, 100);

// ========== 径向渐变 ==========
const radialGradient = ctx.createRadialGradient(100, 100, 10, 100, 100, 80);
radialGradient.addColorStop(0, 'white');
radialGradient.addColorStop(1, 'black');
ctx.fillStyle = radialGradient;
ctx.beginPath();
ctx.arc(100, 100, 80, 0, Math.PI * 2);
ctx.fill();

// ========== 贝塞尔曲线 ==========
ctx.beginPath();
ctx.moveTo(20, 20);
// 二次贝塞尔: (控制点x, 控制点y, 终点x, 终点y)
ctx.quadraticCurveTo(100, 150, 200, 20);
ctx.stroke();

ctx.beginPath();
ctx.moveTo(20, 100);
// 三次贝塞尔: (cp1x, cp1y, cp2x, cp2y, endx, endy)
ctx.bezierCurveTo(50, 0, 150, 200, 200, 100);
ctx.stroke();

// ========== 圆角矩形 ==========
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// ========== 变换 ==========
ctx.save();
ctx.translate(100, 100);  // 移动原点
ctx.rotate(Math.PI / 4);  // 旋转45度
ctx.scale(2, 2);           // 放大2倍
ctx.fillRect(-25, -25, 50, 50);
ctx.restore();             // 恢复状态

// ========== 裁剪 ==========
ctx.beginPath();
ctx.arc(100, 100, 50, 0, Math.PI * 2);
ctx.clip(); // 后续绘制只在圆形区域内显示
ctx.fillRect(0, 0, 200, 200); // 只显示圆形区域
```

## Canvas 性能优化

```javascript
// ========== 离屏 Canvas 缓存 ==========
const offscreen = document.createElement('canvas');
offscreen.width = 200;
offscreen.height = 200;
const offCtx = offscreen.getContext('2d');
// 预先绘制复杂图形
offCtx.fillStyle = 'red';
offCtx.beginPath();
offCtx.arc(100, 100, 80, 0, Math.PI * 2);
offCtx.fill();
// 之后直接绘制缓存图像
ctx.drawImage(offscreen, 50, 50);

// ========== 减少状态变化 ==========
// 不好的做法: 频繁切换 fillStyle
for (let i = 0; i < 1000; i++) {
    ctx.fillStyle = colors[i % colors.length];
    ctx.fillRect(i * 2, 0, 2, 100);
}

// 好的做法: 分批绘制同色图形
for (const color of colors) {
    ctx.fillStyle = color;
    for (let i = 0; i < 1000; i += colors.length) {
        ctx.fillRect(i * 2, 0, 2, 100);
    }
}

// ========== devicePixelRatio 适配 ==========
function setupHiDPI(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return ctx;
}
```

<!-- Converted from: 48_Canvas基础.html -->
