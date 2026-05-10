# 实战 - Tab 切换 (Tabs)

## 项目需求与功能分析

Tab 切换是网页中最常见的导航模式之一，用于在多个内容面板之间切换。本项目实现一个带下划线动画和 URL hash 同步的 Tab 组件。

### 核心功能

- 点击 Tab 标签切换内容
- 下划线滑动动画
- URL hash 同步
- 键盘导航（左右箭头）
- 支持动态增删 Tab

## 完整代码实现

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Tab 切换</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; display: flex; justify-content: center; padding: 40px 20px; }
  .tabs-container { width: 100%; max-width: 700px; background: #fff; border-radius: 12px; box-shadow: 0 2px 15px rgba(0,0,0,0.08); overflow: hidden; }
  .tabs-nav { display: flex; position: relative; border-bottom: 2px solid #eee; background: #fafafa; }
  .tab-btn { padding: 14px 24px; border: none; background: none; font-size: 15px; cursor: pointer; color: #666; transition: color 0.2s; position: relative; z-index: 1; }
  .tab-btn:hover { color: #333; }
  .tab-btn.active { color: #667eea; font-weight: 600; }
  .tab-btn:focus { outline: none; background: rgba(102,126,234,0.05); }
  .tab-indicator { position: absolute; bottom: -2px; height: 3px; background: #667eea; transition: all 0.3s ease; border-radius: 3px 3px 0 0; }
  .tabs-content { padding: 25px; min-height: 200px; }
  .tab-panel { display: none; animation: fadeIn 0.3s ease; }
  .tab-panel.active { display: block; }
  .tab-panel h3 { font-size: 18px; color: #333; margin-bottom: 12px; }
  .tab-panel p { color: #666; line-height: 1.8; font-size: 14px; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>

<div class="tabs-container">
  <div class="tabs-nav" id="tabsNav">
    <button class="tab-btn active" data-tab="html">HTML</button>
    <button class="tab-btn" data-tab="css">CSS</button>
    <button class="tab-btn" data-tab="js">JavaScript</button>
    <button class="tab-btn" data-tab="api">Web API</button>
    <div class="tab-indicator" id="indicator"></div>
  </div>
  <div class="tabs-content">
    <div class="tab-panel active" id="html">
      <h3>HTML</h3>
      <p>HTML (HyperText Markup Language) 是 Web 的基础。它使用标签定义网页的结构：标题、段落、列表、表格、表单等。
      HTML5 增加了语义标签 (header/nav/article/section) 和多媒体支持 (video/audio)。
      Web Components 标准允许创建可复用的自定义元素。</p>
    </div>
    <div class="tab-panel" id="css">
      <h3>CSS</h3>
      <p>CSS 控制网页的视觉呈现。包括 Flexbox 和 Grid 布局系统、响应式设计 (media queries)、
      动画 (keyframes/transition)、以及 CSS 自定义属性 (变量)。现代 CSS 已能实现复杂的视觉效果。</p>
    </div>
    <div class="tab-panel" id="js">
      <h3>JavaScript</h3>
      <p>JavaScript 是 Web 的编程语言。ES6+ 带来了箭头函数、类、模板字符串、解构、模块等特性。
      异步编程从回调发展到 Promise 再到 async/await。JS 已从浏览器扩展到服务端 (Node.js)。</p>
    </div>
    <div class="tab-panel" id="api">
      <h3>Web API</h3>
      <p>浏览器提供丰富的 API: DOM 操作、Fetch/XHR 网络请求、Canvas/WebGL 图形、WebSocket 实时通信、
      Service Worker 离线缓存、Web Storage、Geolocation 地理位置等。这些 API 赋予了 Web 应用强大的能力。</p>
    </div>
  </div>
</div>

<script>
class Tabs {
  constructor(container) {
    this.container = container;
    this.nav = container.querySelector('#tabsNav');
    this.buttons = container.querySelectorAll('.tab-btn');
    this.panels = container.querySelectorAll('.tab-panel');
    this.indicator = container.querySelector('#indicator');
    this.bindEvents();
    this.updateIndicator(this.getActiveButton());
    this.handleHash();
  }

  bindEvents() {
    this.buttons.forEach(btn => {
      btn.addEventListener('click', () => this.switchTo(btn));
    });
    // 键盘导航
    this.nav.addEventListener('keydown', e => {
      const btns = [...this.buttons];
      const idx = btns.indexOf(document.activeElement);
      if (e.key === 'ArrowRight' && idx < btns.length - 1) {
        btns[idx + 1].focus(); this.switchTo(btns[idx + 1]);
      }
      if (e.key === 'ArrowLeft' && idx > 0) {
        btns[idx - 1].focus(); this.switchTo(btns[idx - 1]);
      }
    });
    window.addEventListener('hashchange', () => this.handleHash());
  }

  switchTo(btn) {
    this.buttons.forEach(b => b.classList.remove('active'));
    this.panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = this.container.querySelector(`#${btn.dataset.tab}`);
    if (panel) panel.classList.add('active');
    this.updateIndicator(btn);
    history.replaceState(null, '', `#${btn.dataset.tab}`);
  }

  updateIndicator(btn) {
    if (!btn) return;
    this.indicator.style.left = btn.offsetLeft + 'px';
    this.indicator.style.width = btn.offsetWidth + 'px';
  }

  getActiveButton() {
    return this.container.querySelector('.tab-btn.active');
  }

  handleHash() {
    const hash = location.hash.slice(1);
    if (hash) {
      const btn = this.container.querySelector(`[data-tab="${hash}"]`);
      if (btn) this.switchTo(btn);
    }
  }
}

new Tabs(document.querySelector('.tabs-container'));
</script>
</body>
</html>
```

## 核心技术详解

### 下划线动画

使用绝对定位 + `transition` 实现下划线滑动。点击时更新 `left` 和 `width`。

### URL Hash 同步

切换 Tab 时更新 `location.hash`，监听 `hashchange` 事件实现浏览器前进后退。

### 键盘导航

监听左右箭头键，在 Tab 按钮之间导航。

## 扩展方向

1. **垂直 Tab**：支持左侧垂直布局
2. **可关闭 Tab**：每个 Tab 带关闭按钮
3. **异步加载**：Tab 内容从 API 动态加载
4. **拖拽排序**：拖拽 Tab 标签调整顺序
5. **右键菜单**：关闭其他、关闭全部
6. **路由集成**：与前端路由框架集成
7. **懒渲染**：首次切换到 Tab 时才渲染内容
