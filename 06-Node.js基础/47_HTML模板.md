# HTML模板


## HTML 模板


template 标签、content 片段、cloning、slot 插槽、组成复用。


## HTML Template API


```
// ========== template 标签 ==========


// ========== 使用模板 ==========
const template = document.getElementById('my-tpl');
const clone = template.content.cloneNode(true);
// clone 是 DocumentFragment, 不会出现在 DOM 树

// ========== 填充数据 ==========
clone.querySelector('h3').textContent = '标题';
clone.querySelector('p').textContent = '内容';

// ========== 添加到 DOM ==========
document.body.appendChild(clone);

// ========== 模板优点 ==========
// 1. 内容不可见 (不渲染)
// 2. 无网络请求
// 3. 可克隆复用
// 4. 保持 HTML 结构
```


## 演示：HTML 模板

点击按钮查看


## 什么是 HTML Template

`<template>` 标签定义了一段HTML片段模板，内容不会被浏览器渲染，可以通过JS克隆后插入DOM。是Web Components的基础技术之一。

## 工作机制

- `<template>` 内的内容存在于 `document` 中但不会显示
- 内容中的 `<img>`、`<video>` 等不会发起网络请求
- `<script>` 标签不会执行
- 通过 `template.content.cloneNode(true)` 克隆后才能使用

## 使用场景

1. **列表渲染**：预定义列表项模板，循环克隆填充数据
2. **组件化**：与Shadow DOM结合构建Web Components
3. **动态内容**：服务端返回模板HTML，客户端填充数据
4. **邮件模板**：预定义邮件结构

## 模板嵌套

`<template>` 可以嵌套使用。外层模板克隆后，内层 `<template>` 仍可继续克隆填充。

## 与其他模板方案对比

| 方案 | 特点 |
|------|------|
| `<template>` | 原生API，无依赖，适合简单场景 |
| Handlebars/Mustache | 语法简洁，需引入库 |
| Vue模板 | 响应式绑定，功能强大 |
| JSX | JS中写模板，需编译 |

## 模板与 slot 实战

```javascript
// ========== 列表模板渲染 ==========
function renderList(container, templateSelector, items) {
    const template = document.querySelector(templateSelector);
    const fragment = document.createDocumentFragment();

    items.forEach(item => {
        const clone = template.content.cloneNode(true);
        clone.querySelector('.title').textContent = item.title;
        clone.querySelector('.description').textContent = item.description;
        clone.querySelector('.date').textContent = item.date;
        if (item.image) {
            clone.querySelector('img').src = item.image;
        }
        fragment.appendChild(clone);
    });

    container.appendChild(fragment);
}

// ========== 带 slot 的模板 ==========
// <template id="card-template">
//   <div class="card">
//     <h3><slot name="title">默认标题</slot></h3>
//     <slot></slot>
//   </div>
// </template>

function createCard(title, content) {
    const template = document.getElementById('card-template');
    const clone = template.content.cloneNode(true);
    clone.querySelector('[slot="title"]').textContent = title;
    clone.querySelector('slot:not([name])').assignedSlot.textContent = content;
    return clone;
}

// ========== 动态模板注册 ==========
class TemplateRegistry {
    constructor() {
        this.templates = new Map();
    }

    register(name, html) {
        const template = document.createElement('template');
        template.innerHTML = html;
        this.templates.set(name, template);
    }

    render(name, data = {}) {
        const template = this.templates.get(name);
        if (!template) throw new Error(`模板 ${name} 未注册`);

        const clone = template.content.cloneNode(true);

        // 简易模板引擎: 替换 {{key}}
        const walker = document.createTreeWalker(clone, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
            walker.currentNode.textContent = walker.currentNode.textContent
                .replace(/\{\{(\w+)\}\}/g, (_, key) => data[key] ?? '');
        }

        return clone;
    }
}

// 使用
const registry = new TemplateRegistry();
registry.register('user', `
    <div class="user">
        <span class="name">{{name}}</span>
        <span class="email">{{email}}</span>
    </div>
`);

const el = registry.render('user', { name: 'Alice', email: 'alice@example.com' });
document.body.appendChild(el);
```

<!-- Converted from: 47_HTML模板.html -->
