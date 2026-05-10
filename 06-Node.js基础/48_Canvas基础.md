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


<!-- Converted from: 48_Canvas基础.html -->
