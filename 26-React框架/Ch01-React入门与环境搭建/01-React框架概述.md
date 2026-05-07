# React 框架概述

## 什么是 React

React 是一个用于构建用户界面的 **JavaScript 库**（注意：不是框架），由 Meta（原 Facebook）团队开发和维护。React 的核心理念是**组件化开发**——将复杂的用户界面拆分成一个个独立、可复用的组件，然后像搭积木一样组合起来。

React 专注于视图层（View Layer），只负责 UI 的渲染和更新。路由、状态管理、数据请求等功能需要搭配第三方库使用，这也是它被称为"库"而非"框架"的原因。

## React 的发展历程

| 时间 | 事件 |
|------|------|
| 2011 | Jordan Walke 在 Facebook 内部创建 React 原型 |
| 2013 | React 开源发布，JSX 语法引入 |
| 2015 | React Native 发布，React 生态爆发 |
| 2016 | React 15 发布，性能优化 |
| 2017 | React 16 (Fiber) 架构重写，支持异步渲染 |
| 2019 | React Hooks 发布（16.8），函数组件成为主流 |
| 2020 | React 17 发布，无新特性，专注渐进式升级 |
| 2022 | React 18 发布，并发渲染、Suspense、自动批处理 |
| 2024 | React 19 发布，Server Components、Actions、新 Hooks |

## 核心概念

### 虚拟 DOM（Virtual DOM）

React 引入了**虚拟 DOM** 的概念来提升性能：

```
数据变化 → 生成新的虚拟DOM → Diff算法对比差异 → 只更新真实DOM中变化的部分
```

虚拟 DOM 是一个轻量级的 JavaScript 对象，它是真实 DOM 的映射。当状态发生变化时，React 会：

1. 创建一棵新的虚拟 DOM 树
2. 与旧的虚拟 DOM 树进行对比（Diffing / Reconciliation）
3. 计算出最小的变更集合
4. 将变更批量应用到真实 DOM

**为什么要用虚拟 DOM？** 直接操作 DOM 的代价很高（回流、重绘），而 JavaScript 对象的操作非常快。通过在 JavaScript 层面做计算，减少不必要的 DOM 操作，从而提升性能。

### 声明式 vs 命令式

```js
// 命令式（jQuery 风格）- 描述"怎么做"
const list = document.getElementById('list');
list.innerHTML = '';
data.forEach(item => {
  const li = document.createElement('li');
  li.textContent = item.name;
  list.appendChild(li);
});
```

```jsx
// 声明式（React 风格）- 描述"要什么"
function List({ data }) {
  return (
    <ul>
      {data.map(item => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
}
```

声明式编程让代码更**可预测**、更**易维护**。你只需要描述"界面应该长什么样"，React 负责如何实现。

### 组件化架构

React 的一切都是组件。组件是独立的、可复用的 UI 片段：

```jsx
// 函数组件（推荐）
function Button({ label, onClick }) {
  return <button onClick={onClick}>{label}</button>;
}

// 使用组件
function App() {
  return (
    <div>
      <Button label="保存" onClick={() => console.log('保存')} />
      <Button label="取消" onClick={() => console.log('取消')} />
    </div>
  );
}
```

组件的特点：
- **封装性**：每个组件管理自己的状态和渲染逻辑
- **可复用性**：同一组件可在多处使用
- **可组合性**：小组件组合成大组件
- **可测试性**：每个组件可以独立测试

### 单向数据流

React 采用**单向数据流**（One-Way Data Flow），数据从父组件通过 props 流向子组件：

```
父组件 State → props → 子组件 → props → 孙组件
       ↑
    通过回调函数通知父组件修改
```

这意味着：
- 数据流向清晰，易于追踪
- 子组件不能直接修改父组件的数据
- 调试时可以轻松定位数据变化的来源

## 单页应用（SPA）

React 常用于构建**单页应用**（Single Page Application）。SPA 的特点是：

- 只加载一个 HTML 页面
- 通过 JavaScript 动态替换页面内容
- 用户体验接近原生应用（无页面刷新）
- 需要客户端路由（如 React Router）

## React 的哲学

React 有几个贯穿始终的设计理念：

1. **Learn Once, Write Anywhere** - 学会 React 后，可以开发 Web（React）、移动端（React Native）、VR 应用
2. **Composition over Inheritance** - 组合优于继承，用组件组合而非类继承来复用逻辑
3. **Unidirectional Data Flow** - 单向数据流让状态管理更可预测
4. **Just JavaScript** - React 尽量贴近原生 JavaScript，减少框架特有的概念

## React vs Vue vs Angular 对比

| 特性 | React | Vue | Angular |
|------|-------|-----|---------|
| 类型 | 库 | 渐进式框架 | 完整框架 |
| 语言 | JSX | 模板 / JSX | TypeScript + 模板 |
| 学习曲线 | 中等 | 较低 | 较陡 |
| 数据流 | 单向 | 双向（v-model） | 双向（ngModel） |
| 状态管理 | useState / 外部库 | reactive() / Pinia | RxJS / Signals |
| 生态 | 极丰富 | 丰富 | 内置齐全 |
| 包大小 | 小（核心） | 小 | 大 |
| 社区 | 最大 | 大 | 中等 |
| 适合场景 | 大型 SPA、跨平台 | 中小型项目、快速原型 | 企业级应用 |

## 何时选择 React

React 是一个很好的选择，当你需要：

- 构建复杂的单页应用
- 跨平台开发（Web + 移动端）
- 一个活跃且庞大的生态系统
- 灵活地选择技术栈（路由、状态管理等由你决定）
- 大型团队协作（组件化 + TypeScript 支持良好）

如果你的项目是简单的静态页面或需要"开箱即用"的全功能框架，Vue 或 Angular 可能更合适。

## 小结

React 通过虚拟 DOM、声明式编程、组件化架构和单向数据流，为构建现代用户界面提供了一套高效、可维护的解决方案。虽然它只是一个"库"而非"框架"，但围绕它构建的生态系统已经足够强大，能够支撑从简单页面到复杂企业应用的各种需求。
