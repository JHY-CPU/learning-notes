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


<!-- Converted from: 47_HTML模板.html -->
