# Shadow DOM


## Shadow DOM


attachShadow、mode(open/closed)、样式隔离、:host/slotted。


## Shadow DOM API


```
// ========== 创建 Shadow DOM ==========
const host = document.getElementById('host');
const shadow = host.attachShadow({ mode: 'open' });
// mode: 'open' - 外部可访问 shadowRoot
// mode: 'closed' - 外部不可访问

// ========== 添加内容 ==========
shadow.innerHTML = `



`;

// ========== 事件重定向 ==========
// Shadow DOM 内部事件会冒泡到外部
// event.composedPath() 显示完整路径

// ========== 插槽 Slot ==========
// 默认内容
// 自定义标题
```


## 演示：Shadow DOM

点击按钮查看


## 什么是 Shadow DOM

Shadow DOM 是Web Components的核心技术之一，提供DOM和CSS的封装隔离。外部样式不会影响Shadow DOM内部，内部样式也不会泄漏到外部。

## 核心概念

- **Shadow Host**：承载Shadow DOM的宿主元素
- **Shadow Root**：Shadow DOM的根节点
- **Shadow Tree**：Shadow DOM内部的DOM树
- **Slot（插槽）**：允许外部内容插入Shadow DOM的占位符

## 样式隔离

Shadow DOM内部样式天然隔离。可用 `:host` 选择宿主元素，`::slotted()` 选择插入的slot内容。外部可通过CSS自定义属性（`--custom-prop`）向Shadow DOM传递样式。

## 使用场景

1. **UI组件库**：封装独立组件，防止样式污染
2. **浏览器内置元素**：`<input>`、`<video>` 的内部实现就是Shadow DOM
3. **第三方插件**：嵌入页面但保持独立样式

## 注意事项

- `mode: 'closed'` 下外部无法通过 `el.shadowRoot` 访问，但实际安全意义有限
- 事件会穿过Shadow DOM边界（事件重定向），可用 `event.composedPath()` 查看完整路径
- 全局样式重置（如 `* { margin: 0 }`）不影响Shadow DOM内部
- Shadow DOM 不支持 SSR（服务端渲染），需额外处理

<!-- Converted from: 46_Shadow DOM.html -->
