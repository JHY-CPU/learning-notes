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


<!-- Converted from: 49_Canvas文字图像.html -->
