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


## SVG 基本形状与动画

```javascript
// ========== SVG 基本形状 ==========
const SVG_NS = 'http://www.w3.org/2000/svg';

// 圆形
const circle = document.createElementNS(SVG_NS, 'circle');
circle.setAttribute('cx', '100');
circle.setAttribute('cy', '100');
circle.setAttribute('r', '50');
circle.setAttribute('fill', '#3498db');

// 椭圆
const ellipse = document.createElementNS(SVG_NS, 'ellipse');
ellipse.setAttribute('cx', '100');
ellipse.setAttribute('cy', '100');
ellipse.setAttribute('rx', '80');
ellipse.setAttribute('ry', '40');

// 多边形 (星形)
const polygon = document.createElementNS(SVG_NS, 'polygon');
const points = [];
for (let i = 0; i < 10; i++) {
    const angle = (i * Math.PI / 5) - Math.PI / 2;
    const r = i % 2 === 0 ? 50 : 25;
    points.push(`${100 + r * Math.cos(angle)},${100 + r * Math.sin(angle)}`);
}
polygon.setAttribute('points', points.join(' '));
polygon.setAttribute('fill', '#f39c12');

// 文字
const text = document.createElementNS(SVG_NS, 'text');
text.setAttribute('x', '50');
text.setAttribute('y', '200');
text.setAttribute('font-size', '24');
text.setAttribute('fill', '#333');
text.textContent = 'SVG 文字';

// ========== SVG 动画 (SMIL) ==========
const animatedCircle = document.createElementNS(SVG_NS, 'circle');
animatedCircle.setAttribute('cx', '100');
animatedCircle.setAttribute('cy', '100');
animatedCircle.setAttribute('r', '30');
animatedCircle.setAttribute('fill', '#e74c3c');

// animate 子元素
const animate = document.createElementNS(SVG_NS, 'animate');
animate.setAttribute('attributeName', 'r');
animate.setAttribute('values', '30;50;30');
animate.setAttribute('dur', '2s');
animate.setAttribute('repeatCount', 'indefinite');
animatedCircle.appendChild(animate);

// ========== SVG 渐变 ==========
const defs = document.createElementNS(SVG_NS, 'defs');
const linearGradient = document.createElementNS(SVG_NS, 'linearGradient');
linearGradient.id = 'myGradient';

const stop1 = document.createElementNS(SVG_NS, 'stop');
stop1.setAttribute('offset', '0%');
stop1.setAttribute('stop-color', '#ff0000');

const stop2 = document.createElementNS(SVG_NS, 'stop');
stop2.setAttribute('offset', '100%');
stop2.setAttribute('stop-color', '#0000ff');

linearGradient.appendChild(stop1);
linearGradient.appendChild(stop2);
defs.appendChild(linearGradient);

// 使用渐变
const gradRect = document.createElementNS(SVG_NS, 'rect');
gradRect.setAttribute('fill', 'url(#myGradient)');
```

## SVG vs Canvas 选择

```
// ========== SVG vs Canvas ==========
//
// SVG 优势:
// - 矢量图形，缩放不失真
// - 每个元素都是 DOM 节点，支持事件
// - CSS 可控制样式
// - 适合图标、图表、交互式图形
//
// Canvas 优势:
// - 像素级操作
// - 大量元素时性能更好 (不需要DOM)
// - 适合游戏、视频处理、复杂动画
// - 图像处理和滤镜
//
// 选择建议:
// - 图表 (D3.js) → SVG
// - 地图 → SVG/Canvas 都可
// - 游戏 → Canvas
// - 图标/LOGO → SVG
// - 实时数据可视化 → Canvas
// - 交互式 UI → SVG
```

## SVG 滤镜

```javascript
// ========== SVG 滤镜 ==========
const defs = document.createElementNS(SVG_NS, 'defs');

// 模糊滤镜
const filter = document.createElementNS(SVG_NS, 'filter');
filter.id = 'blur';
const blur = document.createElementNS(SVG_NS, 'feGaussianBlur');
blur.setAttribute('stdDeviation', '3');
filter.appendChild(blur);
defs.appendChild(filter);

// 阴影滤镜
const shadowFilter = document.createElementNS(SVG_NS, 'filter');
shadowFilter.id = 'shadow';
shadowFilter.setAttribute('x', '-20%');
shadowFilter.setAttribute('y', '-20%');
shadowFilter.setAttribute('width', '140%');
shadowFilter.setAttribute('height', '140%');

const offset = document.createElementNS(SVG_NS, 'feOffset');
offset.setAttribute('dx', '3');
offset.setAttribute('dy', '3');
offset.setAttribute('result', 'offset');

const flood = document.createElementNS(SVG_NS, 'feFlood');
flood.setAttribute('flood-color', 'rgba(0,0,0,0.3)');
flood.setAttribute('result', 'color');

const composite = document.createElementNS(SVG_NS, 'feComposite');
composite.setAttribute('in', 'color');
composite.setAttribute('in2', 'offset');
composite.setAttribute('operator', 'in');
composite.setAttribute('result', 'shadow');

const merge = document.createElementNS(SVG_NS, 'feMerge');
const mergeNode1 = document.createElementNS(SVG_NS, 'feMergeNode');
const mergeNode2 = document.createElementNS(SVG_NS, 'feMergeNode');
mergeNode2.setAttribute('in', 'SourceGraphic');
merge.appendChild(mergeNode1);
merge.appendChild(mergeNode2);

shadowFilter.append(offset, flood, composite, merge);
defs.appendChild(shadowFilter);

// 使用: rect.setAttribute('filter', 'url(#shadow)');
```

<!-- Converted from: 51_SVG操作.html -->
