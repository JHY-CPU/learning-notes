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


## 完整自定义组件示例

```javascript
// ========== 卡片组件 ==========
class MyCard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    static get observedAttributes() {
        return ['title', 'image', 'variant'];
    }

    connectedCallback() {
        this.render();
    }

    attributeChangedCallback() {
        if (this.shadowRoot) this.render();
    }

    render() {
        const title = this.getAttribute('title') || '默认标题';
        const image = this.getAttribute('image') || '';
        const variant = this.getAttribute('variant') || 'default';

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                }
                :host(:hover) {
                    transform: translateY(-2px);
                }
                .card-image {
                    width: 100%;
                    height: 200px;
                    object-fit: cover;
                }
                .card-body {
                    padding: 16px;
                }
                h3 { margin: 0 0 8px; }
                ::slotted(.card-description) {
                    color: #666;
                    font-size: 14px;
                }
            </style>
            ${image ? `<img class="card-image" src="${image}" alt="${title}">` : ''}
            <div class="card-body">
                <h3>${title}</h3>
                <slot name="description"></slot>
                <slot></slot>
            </div>
        `;
    }
}

customElements.define('my-card', MyCard);

// 使用:
// <my-card title="文章标题" image="photo.jpg">
//     <p slot="description" class="card-description">描述文字</p>
//     <p>默认插槽内容</p>
// </my-card>

// ========== 表单输入组件 ==========
class MyInput extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    static get observedAttributes() {
        return ['value', 'placeholder', 'type', 'disabled'];
    }

    get value() {
        return this.shadowRoot.querySelector('input').value;
    }

    set value(val) {
        this.shadowRoot.querySelector('input').value = val;
    }

    connectedCallback() {
        this.render();
        this.shadowRoot.querySelector('input').addEventListener('input', (e) => {
            this.dispatchEvent(new CustomEvent('my-input', {
                detail: { value: e.target.value },
                bubbles: true,
            }));
        });
    }

    render() {
        const type = this.getAttribute('type') || 'text';
        const placeholder = this.getAttribute('placeholder') || '';
        const disabled = this.hasAttribute('disabled');

        this.shadowRoot.innerHTML = `
            <style>
                :host { display: inline-block; }
                input {
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 14px;
                    outline: none;
                    transition: border-color 0.2s;
                }
                input:focus { border-color: #3498db; }
                input:disabled { background: #f5f5f5; cursor: not-allowed; }
            </style>
            <input type="${type}" placeholder="${placeholder}" ${disabled ? 'disabled' : ''}>
        `;
    }
}

customElements.define('my-input', MyInput);
```

## 生命周期与最佳实践

- **constructor**: 初始化 Shadow DOM，设置事件监听器
- **connectedCallback**: 组件挂载到 DOM，适合获取数据、启动动画
- **disconnectedCallback**: 清理定时器、事件监听器，防止内存泄漏
- **attributeChangedCallback**: 响应属性变化，重新渲染
- **observedAttributes**: 必须声明要观察的属性，否则回调不会触发

<!-- Converted from: 45_Web Components.html -->
