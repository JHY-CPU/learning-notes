# Web Components


## Web Components


customElements.define、生命周期(connected/disconnected)、attributeChanged。


## Web Components API


```
// ========== 自定义元素 ==========
class MyElement extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    // 观察属性
    static get observedAttributes() {
        return ['name', 'count'];
    }

    // 生命周期
    connectedCallback() { /* 挂载到 DOM */ }
    disconnectedCallback() { /* 从 DOM 移除 */ }
    attributeChangedCallback(name, old, val) { /* 属性变化 */ }
    adoptedCallback() { /* 被移动到新文档 */ }
}

customElements.define('my-element', MyElement);

// ========== 使用 ==========
//

// ========== Shadow DOM ==========
// 样式隔离: :host, ::slotted
// mode: 'open' (外部可访问), 'closed' (外部不可访问)

// ========== HTML Templates ==========
//
```


## 演示：Web Components

点击按钮查看


<!-- Converted from: 45_Web Components.html -->
