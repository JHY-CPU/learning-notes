# MVC与MVVM架构模式


## 一、MVC模式


MVC（Model-View-Controller）将应用分为三个相互协作的组件，是最经典的软件架构模式。


### 1.1 三个组件


| 组件 | 职责 | 特点 |
| --- | --- | --- |
| Model（模型） | 数据和业务逻辑 | 不依赖View和Controller，独立变化 |
| View（视图） | 用户界面展示 | 从Model获取数据显示，不处理业务逻辑 |
| Controller（控制器） | 接收用户输入，协调M和V | 处理用户交互，更新Model，选择View |


### 1.2 数据流向

Model
View
Controller
用户 → Controller → Model → View → 用户
Arrows

### 1.3 JavaScript实现


```
// Model：数据和业务逻辑
class TodoModel {
    constructor() {
        this.todos = [];
        this.listeners = [];
    }
    add(text) {
        this.todos.push({ id: Date.now(), text, done: false });
        this.notify();
    }
    toggle(id) {
        const todo = this.todos.find(t => t.id === id);
        if (todo) todo.done = !todo.done;
        this.notify();
    }
    subscribe(fn) { this.listeners.push(fn); }
    notify() { this.listeners.forEach(fn => fn(this.todos)); }
}

// View：只负责渲染
class TodoView {
    constructor(container) {
        this.container = container;
    }
    render(todos) {
        this.container.innerHTML = todos.map(t =>
            `${t.text}`
        ).join('');
    }
}

// Controller：协调M和V
class TodoController {
    constructor(model, view) {
        this.model = model;
        this.view = view;
        model.subscribe(todos => view.render(todos));
    }
    addTodo(text) { this.model.add(text); }
    toggleTodo(id) { this.model.toggle(id); }
}
```


## 二、MVP模式


MVP（Model-View-Presenter）是MVC的变体，View和Model完全隔离，通过Presenter通信。


### 2.1 与MVC的区别


| 对比 | MVC | MVP |
| --- | --- | --- |
| View与Model | View可直接读取Model | View与Model完全隔离 |
| 通信方式 | Controller协调 | Presenter双向通信 |
| View被动性 | 相对主动 | 完全被动（Passive View） |
| 可测试性 | 中等 | 更好（View可Mock） |


### 2.2 数据流向

用户操作 → View → Presenter → Model


Model → Presenter → View（更新界面）


**View和Model之间没有直接依赖**

## 三、MVVM模式


MVVM（Model-View-ViewModel）通过数据绑定实现View和ViewModel的自动同步，是Vue、Angular等框架的核心。


### 3.1 三个组件


| 组件 | 职责 | 对应框架 |
| --- | --- | --- |
| Model | 数据和业务逻辑 | API服务、数据层 |
| View | 用户界面（HTML模板） | Vue模板、Angular模板 |
| ViewModel | 暴露数据和命令给View，双向绑定 | Vue实例、Angular Component |


### 3.2 核心：数据绑定

View ←──双向数据绑定──→ ViewModel ←──→ Model


View变化 ──自动同步──→ ViewModel数据


ViewModel数据变化 ──自动同步──→ View更新

### 3.3 Vue.js MVVM示例


```
<!-- View：模板 -->
<div id="app">
    <input v-model="message">  <!-- 双向绑定 -->
    <p>{{ message }}</p>       <!-- 单向绑定 -->
    <button @click="reverse">反转</button>
</div>

// ViewModel：Vue实例
const app = Vue.createApp({
    // data暴露给View的数据
    data() {
        return { message: 'Hello MVVM' };
    },
    // methods暴露给View的命令
    methods: {
        reverse() {
            this.message = this.message.split('').reverse().join('');
        }
    }
}).mount('#app');
```


### 3.4 简易双向绑定实现原理


```
// 简化的数据劫持 + 发布订阅
function defineReactive(obj, key, val) {
    const dep = []; // 订阅者列表

    Object.defineProperty(obj, key, {
        get() {
            // 收集依赖（Watcher）
            if (Dep.target) dep.push(Dep.target);
            return val;
        },
        set(newVal) {
            if (newVal === val) return;
            val = newVal;
            // 通知所有订阅者更新
            dep.forEach(watcher => watcher.update());
        }
    });
}

// Vue 3 使用 Proxy 替代 Object.defineProperty
const reactive = (obj) => new Proxy(obj, {
    get(target, key) {
        track(target, key); // 收集依赖
        return target[key];
    },
    set(target, key, value) {
        target[key] = value;
        trigger(target, key); // 触发更新
        return true;
    }
});
```


## 四、MVC / MVP / MVVM 对比


| 特性 | MVC | MVP | MVVM |
| --- | --- | --- | --- |
| View-Model通信 | Controller协调 | Presenter双向 | 数据绑定自动 |
| View被动性 | 较主动 | 被动 | 被动（自动同步） |
| 耦合度 | 较高 | 较低 | 最低 |
| 可测试性 | 中等 | 好 | 好 |
| 适用场景 | 后端框架 | Android开发 | 前端SPA框架 |
| 代表框架 | Spring MVC, Express | Android MVP | Vue, Angular, React+ |


> **Important:** **核心区别：**
>
>
> MVC：用户 → Controller → Model → View
>
>
> MVP：用户 → View → Presenter → Model → Presenter → View
>
>
> MVVM：View ↔（数据绑定）↔ ViewModel ↔ Model


## 五、知识要点总结


1. MVC：Model-View-Controller，最经典的分层架构
2. MVP：View和Model完全隔离，Presenter负责通信
3. MVVM：数据绑定实现View和ViewModel自动同步
4. Vue双向绑定原理：数据劫持 + 发布订阅模式
5. Vue 2用Object.defineProperty，Vue 3用Proxy
6. 选择依据：团队、项目规模、框架生态


<!-- Converted from: 01_MVC与MVVM.html -->
