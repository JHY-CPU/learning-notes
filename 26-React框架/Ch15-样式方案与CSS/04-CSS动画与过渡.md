# CSS 动画与过渡

## 1. CSS Transitions（过渡）

CSS Transition 用于在属性值之间平滑过渡。

### 基础语法

```css
.element {
  transition: property duration timing-function delay;
  /* 例：transition: all 0.3s ease-in-out 0.1s; */
}
```

| 属性 | 说明 | 示例 |
|------|------|------|
| `transition-property` | 过渡的 CSS 属性 | `opacity`, `transform`, `all` |
| `transition-duration` | 持续时间 | `0.3s`, `300ms` |
| `transition-timing-function` | 缓动函数 | `ease`, `ease-in-out`, `cubic-bezier()` |
| `transition-delay` | 延迟时间 | `0.1s` |

### 缓动函数

```css
/* 预设 */
ease          /* 默认，慢-快-慢 */
linear        /* 匀速 */
ease-in       /* 慢-快 */
ease-out      /* 快-慢 */
ease-in-out   /* 慢-快-慢 */

/* 自定义贝塞尔曲线 */
cubic-bezier(0.68, -0.55, 0.265, 1.55)  /* 弹跳效果 */
cubic-bezier(0.25, 0.1, 0.25, 1)         /* 平滑 */

/* 弹性 */
spring(1, 80, 10, 0)  /* framer-motion 支持 */
```

### React 中的使用

```tsx
// CSS 过渡组件
function FadeIn({ children }: { children: React.ReactNode }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    // 延迟触发进入动画
    requestAnimationFrame(() => setVisible(true))
  }, [])

  return (
    <div
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        transition: 'opacity 0.3s ease-out, transform 0.3s ease-out',
      }}
    >
      {children}
    </div>
  )
}

// 用 CSS 类实现
import './transitions.css'

function Toggle() {
  const [open, setOpen] = useState(false)

  return (
    <div>
      <button onClick={() => setOpen(!open)}>Toggle</button>
      <div className={`panel ${open ? 'panel-open' : 'panel-closed'}`}>
        Content
      </div>
    </div>
  )
}
```

```css
/* transitions.css */
.panel {
  overflow: hidden;
  transition: max-height 0.3s ease-in-out, opacity 0.3s ease-in-out;
}

.panel-open {
  max-height: 500px;
  opacity: 1;
}

.panel-closed {
  max-height: 0;
  opacity: 0;
}
```

---

## 2. CSS Keyframe 动画

### 基础语法

```css
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateX(-100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* 或使用百分比 */
@keyframes bounce {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-20px);
  }
}

.element {
  animation: slideIn 0.5s ease-out forwards;
}
```

### animation 属性

```css
.element {
  animation: name duration timing-function delay iteration-count direction fill-mode;
}

/* 各属性 */
animation-name: slideIn;
animation-duration: 0.5s;
animation-timing-function: ease-out;
animation-delay: 0.1s;
animation-iteration-count: infinite;  /* 或 1, 2, 3... */
animation-direction: alternate;        /* normal, reverse, alternate, alternate-reverse */
animation-fill-mode: forwards;         /* none, forwards, backwards, both */
animation-play-state: running;         /* running, paused */
```

### Tailwind 中的动画

```tsx
// 内置动画
<div className="animate-spin">Loading...</div>       /* 旋转 */
<div className="animate-ping">Notification</div>     /* 脉冲 */
<div className="animate-pulse">Placeholder</div>     /* 呼吸 */
<div className="animate-bounce">Arrow</div>          /* 弹跳 */

// 自定义动画（tailwind.config.js）
// theme.extend.animation
// theme.extend.keyframes
```

### React 中的关键帧

```tsx
// 导入 CSS 文件
import './animations.css'

function Spinner() {
  return <div className="spinner" />
}

function Alert({ message }: { message: string }) {
  return (
    <div className="alert-slide-in">
      {message}
    </div>
  )
}
```

```css
/* animations.css */
@keyframes spin {
  to { transform: rotate(360deg); }
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #e0e0e0;
  border-top-color: #007bff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes slideInRight {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.alert-slide-in {
  animation: slideInRight 0.3s ease-out;
}
```

---

## 3. React Transition Group

React Transition Group 提供声明式的 CSS 过渡控制。

### 安装

```bash
npm install react-transition-group
npm install -D @types/react-transition-group
```

### CSSTransition

```tsx
import { CSSTransition } from 'react-transition-group'
import './modal.css'

function Modal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const nodeRef = useRef(null)

  return (
    <CSSTransition
      in={isOpen}
      timeout={300}
      classNames="modal"
      unmountOnExit
      nodeRef={nodeRef}
    >
      <div ref={nodeRef} className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
          <h2>Modal Title</h2>
          <p>Modal content goes here</p>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </CSSTransition>
  )
}
```

```css
/* modal.css */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 进入动画 */
.modal-enter {
  opacity: 0;
}
.modal-enter-active {
  opacity: 1;
  transition: opacity 300ms ease-out;
}

/* 退出动画 */
.modal-exit {
  opacity: 1;
}
.modal-exit-active {
  opacity: 0;
  transition: opacity 300ms ease-out;
}
```

### TransitionGroup（列表动画）

```tsx
import { TransitionGroup, CSSTransition } from 'react-transition-group'

function TodoList() {
  const [todos, setTodos] = useState([
    { id: 1, text: 'Learn React' },
    { id: 2, text: 'Learn TypeScript' },
  ])

  return (
    <TransitionGroup component="ul" className="todo-list">
      {todos.map(todo => (
        <CSSTransition
          key={todo.id}
          timeout={300}
          classNames="todo"
          nodeRef={useRef(null)}
        >
          <li ref={useRef(null)}>
            {todo.text}
            <button onClick={() => setTodos(t => t.filter(x => x.id !== todo.id))}>
              Delete
            </button>
          </li>
        </CSSTransition>
      ))}
    </TransitionGroup>
  )
}
```

```css
.todo-enter {
  opacity: 0;
  transform: translateX(-30px);
}
.todo-enter-active {
  opacity: 1;
  transform: translateX(0);
  transition: all 300ms ease-out;
}
.todo-exit {
  opacity: 1;
  transform: translateX(0);
}
.todo-exit-active {
  opacity: 0;
  transform: translateX(30px);
  transition: all 300ms ease-out;
}
```

---

## 4. Framer Motion 基础

Framer Motion 是 React 中最流行的动画库，提供声明式、物理驱动的动画。

### 安装

```bash
npm install framer-motion
```

### 基础动画

```tsx
import { motion } from 'framer-motion'

function App() {
  return (
    <div>
      {/* 淡入动画 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Hello
      </motion.div>

      {/* 弹跳效果 */}
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{
          type: 'spring',
          stiffness: 260,
          damping: 20
        }}
      >
        Bouncy
      </motion.div>

      {/* 旋转动画 */}
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        Spinning
      </motion.div>
    </div>
  )
}
```

### 交互动画

```tsx
function InteractiveButton() {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="px-6 py-3 bg-blue-600 text-white rounded-lg"
    >
      Click me
    </motion.button>
  )
}

function DragExample() {
  return (
    <motion.div
      drag
      dragConstraints={{ left: -100, right: 100, top: -100, bottom: 100 }}
      dragElastic={0.2}
      whileDrag={{ scale: 1.1, boxShadow: '0 10px 25px rgba(0,0,0,0.2)' }}
      className="w-20 h-20 bg-blue-500 rounded-lg cursor-grab"
    >
      Drag me
    </motion.div>
  )
}
```

### 变体（Variants）

```tsx
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,  // 子元素依次动画，间隔 0.1s
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
}

function StaggeredList({ items }: { items: string[] }) {
  return (
    <motion.ul
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-2"
    >
      {items.map((item, i) => (
        <motion.li
          key={i}
          variants={itemVariants}
          className="p-4 bg-white rounded-lg shadow"
        >
          {item}
        </motion.li>
      ))}
    </motion.ul>
  )
}
```

---

## 5. AnimatePresence（退出动画）

AnimatePresence 使组件在卸载时也能播放动画。

### 基础用法

```tsx
import { motion, AnimatePresence } from 'framer-motion'

function App() {
  const [show, setShow] = useState(true)

  return (
    <div>
      <button onClick={() => setShow(!show)}>Toggle</button>

      <AnimatePresence>
        {show && (
          <motion.div
            key="modal"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 flex items-center justify-center bg-black/50"
          >
            <div className="bg-white p-8 rounded-lg">
              <h2>Modal</h2>
              <button onClick={() => setShow(false)}>Close</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
```

### 路由切换动画

```tsx
import { AnimatePresence, motion } from 'framer-motion'
import { useLocation, Routes, Route } from 'react-router-dom'

function AnimatedRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.2 }}
      >
        <Routes location={location}>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  )
}
```

### 列表增删动画

```tsx
function AnimatedList() {
  const [items, setItems] = useState([
    { id: 1, text: 'Item 1' },
    { id: 2, text: 'Item 2' },
  ])

  return (
    <ul>
      <AnimatePresence>
        {items.map(item => (
          <motion.li
            key={item.id}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="flex items-center justify-between p-4 bg-white mb-2 rounded-lg"
          >
            <span>{item.text}</span>
            <button onClick={() => setItems(items.filter(i => i.id !== item.id))}>
              X
            </button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
```

---

## 6. 布局动画

使用 `layout` 属性实现布局变化的平滑动画：

```tsx
function ExpandableCard() {
  const [expanded, setExpanded] = useState(false)

  return (
    <motion.div
      layout
      onClick={() => setExpanded(!expanded)}
      className={`bg-white rounded-lg shadow cursor-pointer p-4 ${
        expanded ? 'w-96' : 'w-48'
      }`}
    >
      <motion.h2 layout>Card Title</motion.h2>
      <AnimatePresence>
        {expanded && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            Extended content here...
          </motion.p>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// layoutId — 跨组件布局动画
function ImageGallery({ images }: { images: Image[] }) {
  const [selected, setSelected] = useState<Image | null>(null)

  return (
    <>
      <div className="grid grid-cols-3 gap-4">
        {images.map(img => (
          <motion.div
            key={img.id}
            layoutId={`image-${img.id}`}
            onClick={() => setSelected(img)}
            className="cursor-pointer"
          >
            <motion.img src={img.thumb} layout />
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {selected && (
          <motion.div
            layoutId={`image-${selected.id}`}
            className="fixed inset-0 flex items-center justify-center"
          >
            <motion.img src={selected.full} />
            <button onClick={() => setSelected(null)}>Close</button>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
```

---

## 7. 手势驱动动画

```tsx
function SwipeToDelete({ children, onDelete }: {
  children: React.ReactNode
  onDelete: () => void
}) {
  const x = useMotionValue(0)
  const opacity = useTransform(x, [-200, 0], [0, 1])
  const background = useTransform(
    x,
    [-200, 0],
    ['#ef4444', '#ffffff']
  )

  return (
    <div className="relative overflow-hidden">
      {/* 背景删除按钮 */}
      <motion.div
        className="absolute inset-0 flex items-center justify-end pr-4 bg-red-500"
        style={{ opacity: useTransform(x, [-200, -100], [1, 0]) }}
      >
        <span className="text-white font-bold">删除</span>
      </motion.div>

      {/* 可滑动内容 */}
      <motion.div
        drag="x"
        dragConstraints={{ left: -200, right: 0 }}
        dragElastic={0.1}
        style={{ x, opacity, background }}
        onDragEnd={(_, info) => {
          if (info.offset.x < -150) {
            onDelete()
          }
        }}
        className="relative bg-white p-4"
      >
        {children}
      </motion.div>
    </div>
  )
}
```

---

## 8. Spring 物理动画

```tsx
import { motion, useSpring, useMotionValue, useTransform } from 'framer-motion'

function SpringCounter() {
  const count = useMotionValue(0)
  const rounded = useTransform(count, Math.round)
  const springValue = useSpring(count, {
    stiffness: 100,
    damping: 30
  })

  const [displayCount, setDisplayCount] = useState(0)

  useEffect(() => {
    springValue.on('change', (v) => {
      setDisplayCount(Math.round(v))
    })
  }, [springValue])

  return (
    <div>
      <span className="text-4xl font-bold">{displayCount}</span>
      <button onClick={() => count.set(count.get() + 1)}>+1</button>
      <button onClick={() => count.set(count.get() - 1)}>-1</button>
    </div>
  )
}

// 弹性按钮
function SpringButton({ children }: { children: React.ReactNode }) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{
        type: 'spring',
        stiffness: 400,
        damping: 17
      }}
    >
      {children}
    </motion.button>
  )
}
```

---

## 9. CSS 动画 vs JS 动画

### 选择指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 简单 hover/focus 效果 | **CSS transition** | 零 JS 开销，浏览器优化 |
| 加载动画、指示器 | **CSS keyframes** | 轻量，独立于 React |
| 页面进入/退出 | **Framer Motion** | 声明式，易维护 |
| 手势交互 | **Framer Motion** | 物理引擎，响应好 |
| 列表排序动画 | **Framer Motion layout** | 自动计算差异 |
| 大量元素动画 | **CSS animations** | 不阻塞主线程 |
| SVG 路径动画 | **Framer Motion / CSS** | 两者都好 |

### 性能建议

```css
/* ✅ 只动画这些属性（GPU 加速，不触发重排） */
transform: translate/scale/rotate/skew
opacity

/* ❌ 避免动画这些属性（触发重排，性能差） */
width, height
top, left, right, bottom
margin, padding
font-size
```

```tsx
// 使用 will-change 提示浏览器优化
<motion.div
  style={{ willChange: 'transform, opacity' }}
  animate={{ x: 100, opacity: 1 }}
>
  Content
</motion.div>

// 但不要过度使用 will-change
// 只在动画开始前添加，结束后移除
```

---

## 10. 实战：Toast 通知组件

```tsx
import { motion, AnimatePresence } from 'framer-motion'

interface Toast {
  id: string
  message: string
  type: 'success' | 'error' | 'info'
}

function ToastContainer({ toasts, onRemove }: {
  toasts: Toast[]
  onRemove: (id: string) => void
}) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      <AnimatePresence>
        {toasts.map(toast => (
          <motion.div
            key={toast.id}
            initial={{ opacity: 0, x: 100, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.9 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            layout
            className={`
              px-4 py-3 rounded-lg shadow-lg cursor-pointer
              ${toast.type === 'success' ? 'bg-green-500 text-white' : ''}
              ${toast.type === 'error' ? 'bg-red-500 text-white' : ''}
              ${toast.type === 'info' ? 'bg-blue-500 text-white' : ''}
            `}
            onClick={() => onRemove(toast.id)}
          >
            {toast.message}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}
```

---

## 总结

- **CSS Transition**：简单属性变化，hover/focus 效果
- **CSS Keyframes**：循环动画、加载指示器
- **React Transition Group**：声明式 CSS 过渡控制
- **Framer Motion**：功能最全，支持手势、物理动画、布局动画
- **AnimatePresence**：解决退出动画问题
- 只动画 `transform` 和 `opacity` 以保证性能
- 用 `layout` 属性实现布局变化的平滑过渡
