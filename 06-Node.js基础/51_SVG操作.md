# SVG操作


## SVG 操作


SVG 命名空间、createElementNS、属性操作、交互事件、动画。


## SVG DOM API


```
// ========== SVG 命名空间 ==========
const SVG_NS = 'http://www.w3.org/2000/svg';

// ========== 创建 SVG 元素 ==========
const svg = document.createElementNS(SVG_NS, 'svg');
svg.setAttribute('width', '400');
svg.setAttribute('height', '300');

// ========== 创建图形 ==========
const rect = document.createElementNS(SVG_NS, 'rect');
rect.setAttribute('x', '10');
rect.setAttribute('y', '10');
rect.setAttribute('width', '100');
rect.setAttribute('height', '50');
rect.setAttribute('fill', '#3498db');

// ========== 基本形状 ==========
//

// ========== 路径 ==========
// M = moveto, L = lineto, H = horizontal, V = vertical
// C = curveto, S = smooth curveto, Q = quadratic, T = smooth quadratic
// A = arc, Z = closepath

// ========== 交互事件 ==========
rect.addEventListener('click', () => {
    rect.setAttribute('fill', '#e74c3c');
});
```


## 演示：SVG 操作

点击按钮查看


<!-- Converted from: 51_SVG操作.html -->
