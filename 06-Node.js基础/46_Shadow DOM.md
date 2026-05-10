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


<!-- Converted from: 46_Shadow DOM.html -->
