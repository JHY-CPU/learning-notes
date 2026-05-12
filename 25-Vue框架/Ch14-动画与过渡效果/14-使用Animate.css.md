# 使用 Animate.css

## 一、概念说明

Animate.css 是一个流行的 CSS 动画库，提供了大量预定义的动画效果（如弹跳、淡入、翻转等）。在 Vue 中使用时，通过自定义过渡类名将 Animate.css 类名映射到 `<Transition>` 组件上。

```vue
<script setup>
import { ref } from 'vue'
import 'animate.css'

const show = ref(true)
</script>

<template>
  <button @click="show = !show">切换</button>
  <Transition
    enter-active-class="animate__animated animate__bounceIn"
    leave-active-class="animate__animated animate__bounceOut"
  >
    <div v-if="show" class="box">Animate.css 效果</div>
  </Transition>
</template>
```

## 二、具体用法

### 2.1 安装与引入

```bash
npm install animate.css
```

```js
// main.js 或组件中
import 'animate.css'
```

### 2.2 常用动画类名

| 效果 | 进入 | 离开 |
|------|------|------|
| 弹跳 | `animate__bounceIn` | `animate__bounceOut` |
| 淡入 | `animate__fadeIn` | `animate__fadeOut` |
| 滑入 | `animate__slideInLeft` | `animate__slideOutRight` |
| 翻转 | `animate__flipInX` | `animate__flipOutX` |
| 缩放 | `animate__zoomIn` | `animate__zoomOut` |

### 2.3 自定义动画时长

```css
/* 覆盖默认时长 */
.animate__animated { --animate-duration: 0.3s; }
```

## 四、动态选择动画效果

```vue
<script setup>
import { ref } from 'vue'
import 'animate.css'

const show = ref(true)
const enterAnim = ref('fadeIn')
const leaveAnim = ref('fadeOut')

const animations = [
  { label: '淡入淡出', enter: 'fadeIn', leave: 'fadeOut' },
  { label: '弹跳', enter: 'bounceIn', leave: 'bounceOut' },
  { label: '滑入', enter: 'slideInLeft', leave: 'slideOutRight' },
  { label: '缩放', enter: 'zoomIn', leave: 'zoomOut' },
  { label: '翻转', enter: 'flipInX', leave: 'flipOutX' }
]
</script>

<template>
  <div>
    <select v-for="anim in animations" :key="anim.label">
      <option @click="enterAnim = anim.enter; leaveAnim = anim.leave">
        {{ anim.label }}
      </option>
    </select>
    <button @click="show = !show">切换</button>
  </div>

  <Transition
    :enter-active-class="`animate__animated animate__${enterAnim}`"
    :leave-active-class="`animate__animated animate__${leaveAnim}`"
  >
    <div v-if="show" class="box">Animate.css 效果</div>
  </Transition>
</template>
```

## 五、自定义动画时长与延迟

```css
/* 全局覆盖 */
:root {
  --animate-duration: 0.5s;
  --animate-delay: 0.2s;
}

/* 组件级别覆盖 */
.my-box.animate__animated {
  --animate-duration: 0.3s;
  --animate-delay: 0.1s;
}

/* 特定动画覆盖 */
.animate__bounceIn {
  --animate-duration: 0.8s;
}
```

## 六、TransitionGroup 配合 Animate.css

```vue
<template>
  <TransitionGroup
    tag="ul"
    enter-active-class="animate__animated animate__fadeInUp"
    leave-active-class="animate__animated animate__fadeOutDown"
  >
    <li v-for="item in items" :key="item.id">
      {{ item.text }}
    </li>
  </TransitionGroup>
</template>
```

## 三、注意事项与常见陷阱

- 必须同时包含 `animate__animated` 基础类和具体动画类
- `enter-active-class` 和 `leave-active-class` 中的类名用**空格**分隔
- Animate.css v4+ 使用 `animate__` 前缀，v3 使用 `animated` 无前缀
- 自定义过渡类名会覆盖 `name` 属性的默认类名
- Animate.css 的移动类名不参与 `TransitionGroup` 的 `move` 过渡
