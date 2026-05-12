# v-on 详解

## 一、概念说明
`v-on` 用于**监听 DOM 事件**并在触发时执行 JavaScript 表达式。缩写为 `@`。支持事件修饰符和按键修饰符。

## 二、具体用法

### 2.1 基本用法
```vue
<template>
  <!-- 完整写法 -->
  <button v-on:click="count++">+1</button>

  <!-- 缩写 -->
  <button @click="count++">+1</button>

  <!-- 调用方法 -->
  <button @click="handleClick">点击</button>

  <!-- 传递参数 -->
  <button @click="handleClick('hello', $event)">带参数</button>
</template>
<script setup>
import { ref } from 'vue'
const count = ref(0)

function handleClick(msg, event) {
  console.log(msg, event.target)
}
</script>
```

### 2.2 事件修饰符
```vue
<template>
  <!-- 阻止冒泡 -->
  <div @click.stop="handleClick">阻止冒泡</div>

  <!-- 阻止默认行为 -->
  <form @submit.prevent="onSubmit">阻止默认</form>

  <!-- 仅触发一次 -->
  <button @click.once="handleOnce">仅一次</button>

  <!-- 串联修饰符 -->
  <a @click.stop.prevent="handle">阻止冒泡和默认</a>
</template>
```

### 2.3 按键/鼠标修饰符
```vue
<template>
  <!-- 按键修饰符 -->
  <input @keyup.enter="submit" />
  <input @keyup.esc="cancel" />

  <!-- 鼠标修饰符 -->
  <div @click.left="leftClick">左键</div>
  <div @click.right="rightClick">右键</div>

  <!-- 系统修饰符 -->
  <div @click.ctrl="ctrlClick">Ctrl+点击</div>
</template>
```

### 2.4 动态事件名
```vue
<template>
  <button @[event]="handler">动态事件</button>
</template>
<script setup>
import { ref } from 'vue'
const event = ref('click')  // 可切换为 'mouseover' 等
</script>
```

## 三、注意事项与常见陷阱
- 事件处理函数中可以访问 `$event` 获取原生事件对象
- 修饰符可以链式使用：`@click.stop.prevent`
- 不要在模板中使用复杂表达式，应抽离为方法
- `@click.self` 只在元素本身（非子元素）触发时执行

## 四、事件修饰符完整列表

```
.stop       → 阻止事件冒泡 (event.stopPropagation())
.prevent    → 阻止默认行为 (event.preventDefault())
.capture    → 使用事件捕获模式
.self       → 只在事件目标是元素自身时触发
.once       → 事件只触发一次
.passive    → 不阻止默认行为（提升滚动性能）
```

### 4.1 修饰符组合与顺序
```vue
<template>
  <!-- 顺序很重要 -->
  <a @click.prevent.stop="handle">阻止默认 + 阻止冒泡</a>

  <!-- .self 和 .stop 的区别 -->
  <div @click.self="handle">
    <!-- 点击子元素不会触发，但冒泡不会被阻止 -->
    <span>点击这里不触发父元素的 handle</span>
  </div>
</template>
```

### 4.2 .passive 提升滚动性能
```vue
<template>
  <!-- passive: 不调用 preventDefault()，提升滚动流畅度 -->
  <div @scroll.passive="onScroll">
    <!-- 大量内容 -->
  </div>

  <!-- ⚠️ 不要同时使用 .passive 和 .prevent -->
  <!-- <div @scroll.passive.prevent="onScroll"> 会报错 -->
</template>
```

## 五、按键修饰符

```vue
<template>
  <!-- 常用按键 -->
  <input @keyup.enter="submit" />
  <input @keyup.esc="cancel" />
  <input @keyup.tab="onTab" />
  <input @keyup.delete="onDelete" />
  <input @keyup.space="onSpace" />
  <input @keyup.up="onUp" />
  <input @keyup.down="onDown" />

  <!-- 组合键 -->
  <input @keyup.ctrl.enter="ctrlEnter" />
  <input @keyup.alt.a="onAltA" />
  <input @keyup.shift.delete="onShiftDelete" />

  <!-- 精确修饰符：只在精确匹配时触发 -->
  <input @keyup.ctrl.exact="onlyCtrl" />
  <input @click.ctrl.exact="onlyCtrlClick" />
  <input @click.exact="onlyClick" />  <!-- 无任何修饰键时触发 -->
</template>
```

## 六、自定义按键修饰符

```js
// main.js
// Vue 3 中使用 key 名替代 keyCode
// 自定义按键别名
app.config.keyCodes = {
  f1: 112,
  mediaPlayPause: 179
}
```

## 七、鼠标按钮修饰符

```vue
<template>
  <div @click.left="leftClick">左键</div>
  <div @click.right="rightClick">右键</div>
  <div @click.middle="middleClick">中键</div>
</template>
```
