# TailwindCSS 与 React

## 1. 安装与配置

### Vite + React + Tailwind CSS

```bash
npm create vite@latest my-app -- --template react-ts
cd my-app
npm install -D tailwindcss @tailwindcss/vite
```

### Vite 配置

```ts
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

### 入口 CSS 文件

```css
/* src/index.css */
@import "tailwindcss";

/* 自定义基础样式 */
@layer base {
  html {
    scroll-behavior: smooth;
  }

  body {
    @apply text-gray-900 bg-white;
  }
}
```

```tsx
// main.tsx
import './index.css'
import App from './App'

// App.tsx 中即可使用 Tailwind 类名
```

---

## 2. Utility-First 原子化方法

Tailwind 使用原子化 CSS 类，每个类对应一个 CSS 属性。

### 常用工具类速查

```tsx
// 布局
<div className="flex items-center justify-between">...</div>
<div className="grid grid-cols-3 gap-4">...</div>
<div className="container mx-auto px-4">...</div>

// 间距
<div className="p-4 m-2 px-6 py-3">...</div>        // padding/margin
<div className="space-x-4 space-y-2">...</div>       // 子元素间距

// 尺寸
<div className="w-full h-screen max-w-md min-h-0">...</div>
<div className="w-1/2 h-64">...</div>

// 颜色
<div className="bg-blue-500 text-white">...</div>
<div className="bg-gray-100 text-gray-900">...</div>
<div className="border border-gray-300">...</div>

// 排版
<p className="text-sm font-bold leading-relaxed">...</p>
<p className="text-lg font-medium tracking-wide">...</p>
<p className="text-center text-justify uppercase">...</p>

// 圆角与阴影
<div className="rounded-lg shadow-md">...</div>
<div className="rounded-full shadow-xl">...</div>

// 过渡与动画
<button className="transition duration-200 ease-in-out hover:bg-blue-600">
  Click
</button>
```

### 颜色调色板

```
gray-50 ~ gray-950   (灰)
red-50 ~ red-950     (红)
blue-50 ~ blue-950   (蓝)
green-50 ~ green-950 (绿)
yellow-50 ~ yellow-950 (黄)
purple-50 ~ purple-950 (紫)
```

---

## 3. 响应式设计

### 断点系统

| 前缀 | 最小宽度 | 对应 |
|------|---------|------|
| `sm:` | 640px | 手机横屏 |
| `md:` | 768px | 平板 |
| `lg:` | 1024px | 小桌面 |
| `xl:` | 1280px | 桌面 |
| `2xl:` | 1536px | 大桌面 |

```tsx
// 移动优先：小屏幕样式是基础，大屏幕用前缀覆盖
function ResponsiveGrid() {
  return (
    <div className="
      grid
      grid-cols-1              /* 手机：1列 */
      sm:grid-cols-2           /* 小屏幕：2列 */
      md:grid-cols-3           /* 平板：3列 */
      lg:grid-cols-4           /* 桌面：4列 */
      gap-4 p-4
    ">
      <div className="bg-white rounded-lg shadow p-6">Card 1</div>
      <div className="bg-white rounded-lg shadow p-6">Card 2</div>
      <div className="bg-white rounded-lg shadow p-6">Card 3</div>
      <div className="bg-white rounded-lg shadow p-6">Card 4</div>
    </div>
  )
}

// 响应式文字大小
<h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl">
  响应式标题
</h1>

// 响应式显示/隐藏
<nav className="hidden md:flex">
  {/* 桌面端显示 */}
</nav>
<nav className="md:hidden">
  {/* 移动端显示 */}
</nav>
```

---

## 4. 暗色模式

```tsx
// tailwind.config.js（v3）
export default {
  darkMode: 'class',  // 使用 class 策略
  // ...
}
```

```html
<!-- 通过 class 切换 -->
<html class="dark">
  <body>...</body>
</html>
```

```tsx
// ThemeToggle.tsx
function ThemeToggle() {
  const [dark, setDark] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  return (
    <button
      onClick={() => setDark(!dark)}
      className="p-2 rounded-lg bg-gray-200 dark:bg-gray-800"
    >
      {dark ? '☀️' : '🌙'}
    </button>
  )
}

// 暗色模式样式
function Card() {
  return (
    <div className="
      bg-white dark:bg-gray-800
      text-gray-900 dark:text-gray-100
      border border-gray-200 dark:border-gray-700
      shadow-md dark:shadow-lg
      rounded-lg p-6
    ">
      <h2 className="text-xl font-bold">Title</h2>
      <p className="text-gray-600 dark:text-gray-400 mt-2">
        Content with dark mode support
      </p>
    </div>
  )
}
```

---

## 5. @apply 指令

将多个工具类组合为一个可复用的 CSS 类：

```css
/* styles.css */
@layer components {
  /* 按钮基础 */
  .btn {
    @apply inline-flex items-center justify-center
           px-4 py-2 rounded-lg font-medium
           transition-colors duration-200
           focus:outline-none focus:ring-2 focus:ring-offset-2;
  }

  .btn-primary {
    @apply btn bg-blue-600 text-white
           hover:bg-blue-700
           focus:ring-blue-500;
  }

  .btn-secondary {
    @apply btn bg-gray-600 text-white
           hover:bg-gray-700
           focus:ring-gray-500;
  }

  .btn-danger {
    @apply btn bg-red-600 text-white
           hover:bg-red-700
           focus:ring-red-500;
  }

  /* 输入框 */
  .input {
    @apply block w-full px-3 py-2
           border border-gray-300 rounded-lg
           text-gray-900 placeholder-gray-400
           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
           dark:bg-gray-800 dark:border-gray-600 dark:text-gray-100;
  }

  /* 卡片 */
  .card {
    @apply bg-white dark:bg-gray-800
           rounded-xl shadow-md
           p-6 border border-gray-200 dark:border-gray-700;
  }
}
```

### 在 React 中使用

```tsx
// 直接使用类名
<button className="btn-primary">Click me</button>
<input className="input" placeholder="Search..." />

// 结合 clsx 动态切换
import clsx from 'clsx'

function Button({ variant = 'primary', className, ...props }) {
  return (
    <button
      className={clsx(
        `btn-${variant}`,
        className
      )}
      {...props}
    />
  )
}
```

> **注意**：`@apply` 应适度使用。过多使用会失去 Tailwind 原子化的优势。如果发现大量使用 `@apply`，考虑改用组件抽象。

---

## 6. 自定义主题

### tailwind.config.js

```ts
// tailwind.config.ts
import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      // 自定义颜色
      colors: {
        brand: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      // 自定义字体
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      // 自定义间距
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
      // 自定义阴影
      boxShadow: {
        'glow': '0 0 15px rgba(59, 130, 246, 0.5)',
        'inner-lg': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.1)',
      },
      // 自定义动画
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'bounce-slow': 'bounce 2s infinite',
        'fade-in': 'fadeIn 0.5s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      // 自定义断点（扩展而非覆盖）
      screens: {
        'xs': '475px',
        '3xl': '1920px',
      },
    },
  },
  plugins: [],
} satisfies Config
```

---

## 7. JIT 模式（Just-In-Time）

Tailwind CSS v3 默认启用 JIT 模式，按需生成 CSS。

### JIT 优势

```tsx
// 可以使用任意值
<div className="top-[117px]">...</div>         // top: 117px
<div className="bg-[#1da1f2]">...</div>         // background: #1da1f2
<div className="grid-cols-[1fr_2fr_1fr]">...</div>
<div className="text-[22px]">...</div>
<div className="shadow-[0_5px_15px_rgba(0,0,0,0.3)]">...</div>

// 任意属性
<div className="[mask-type:luminance]">...</div>
<div className="[&:nth-child(3)]:text-red-500">...</div>

// 变体嵌套
<div className="dark:md:hover:bg-gray-800">...</div>
```

---

## 8. 类名排序

### prettier-plugin-tailwindcss

自动排序 Tailwind 类名，保持一致性：

```bash
npm install -D prettier prettier-plugin-tailwindcss
```

```json
// .prettierrc
{
  "plugins": ["prettier-plugin-tailwindcss"]
}
```

排序前后对比：
```tsx
// 排序前
<div className="p-4 flex bg-white items-center rounded-lg shadow hover:bg-gray-50">

// 排序后（自动）
<div className="flex items-center rounded-lg bg-white p-4 shadow hover:bg-gray-50">
```

排序规则：布局 → 定位 → 显示 → 弹性/网格 → 间距 → 尺寸 → 排版 → 背景 → 边框 → 效果 → 交互。

---

## 9. Headless UI 集成

Headless UI 提供无样式的交互组件，配合 Tailwind 使用：

```bash
npm install @headlessui/react
```

```tsx
import { Menu, Transition } from '@headlessui/react'

function Dropdown() {
  return (
    <Menu as="div" className="relative inline-block text-left">
      <Menu.Button className="inline-flex w-full justify-center rounded-lg
        bg-blue-600 px-4 py-2 text-sm font-medium text-white
        hover:bg-blue-700 focus:outline-none focus-visible:ring-2
        focus-visible:ring-white focus-visible:ring-opacity-75">
        Options
      </Menu.Button>

      <Transition
        enter="transition ease-out duration-100"
        enterFrom="transform opacity-0 scale-95"
        enterTo="transform opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="transform opacity-100 scale-100"
        leaveTo="transform opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right
          divide-y divide-gray-100 rounded-md bg-white shadow-lg
          ring-1 ring-black ring-opacity-5 focus:outline-none">
          <div className="px-1 py-1">
            <Menu.Item>
              {({ active }) => (
                <button className={`${
                  active ? 'bg-blue-500 text-white' : 'text-gray-900'
                } group flex w-full items-center rounded-md px-2 py-2 text-sm`}>
                  Edit
                </button>
              )}
            </Menu.Item>
          </div>
        </Menu.Items>
      </Transition>
    </Menu>
  )
}
```

---

## 10. 组件模式

### 可复用按钮组件

```tsx
import { cva, type VariantProps } from 'class-variance-authority'
import clsx from 'clsx'

// 使用 class-variance-authority 管理变体
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none',
  {
    variants: {
      variant: {
        primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
        secondary: 'bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500',
        danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
        ghost: 'bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800',
        outline: 'border-2 border-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800',
      },
      size: {
        sm: 'px-3 py-1.5 text-sm',
        md: 'px-4 py-2 text-sm',
        lg: 'px-6 py-3 text-base',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
)

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean
}

function Button({
  className,
  variant,
  size,
  loading,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(buttonVariants({ variant, size }), className)}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {children}
    </button>
  )
}
```

### Tailwind 与组件抽象

```tsx
// ✅ 好的抽象：封装为组件
function Badge({
  children,
  color = 'blue'
}: {
  children: React.ReactNode
  color?: 'blue' | 'green' | 'red' | 'yellow'
}) {
  const colors = {
    blue: 'bg-blue-100 text-blue-800',
    green: 'bg-green-100 text-green-800',
    red: 'bg-red-100 text-red-800',
    yellow: 'bg-yellow-100 text-yellow-800',
  }

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  )
}

// 使用
<Badge color="green">Active</Badge>
<Badge color="red">Inactive</Badge>
```

---

## 11. 常用组件 Tailwind 实现

### 表单

```tsx
function FormField({
  label,
  error,
  ...inputProps
}: {
  label: string
  error?: string
} & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        {label}
      </label>
      <input
        className={`
          block w-full px-3 py-2 rounded-lg border
          text-gray-900 dark:text-gray-100
          placeholder-gray-400 dark:placeholder-gray-500
          bg-white dark:bg-gray-800
          focus:outline-none focus:ring-2 focus:ring-offset-0
          transition-colors
          ${error
            ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
            : 'border-gray-300 dark:border-gray-600 focus:ring-blue-500 focus:border-blue-500'
          }
        `}
        {...inputProps}
      />
      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  )
}
```

### 导航栏

```tsx
function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <nav className="bg-white dark:bg-gray-900 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <span className="text-xl font-bold text-gray-900 dark:text-white">
              Logo
            </span>
          </div>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center space-x-8">
            <a href="#" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Home</a>
            <a href="#" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">About</a>
            <a href="#" className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">Contact</a>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={() => setOpen(!open)}
            className="md:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            {open ? '✕' : '☰'}
          </button>
        </div>

        {/* Mobile nav */}
        {open && (
          <div className="md:hidden py-4 space-y-2">
            <a href="#" className="block px-3 py-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800">Home</a>
            <a href="#" className="block px-3 py-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800">About</a>
            <a href="#" className="block px-3 py-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800">Contact</a>
          </div>
        )}
      </div>
    </nav>
  )
}
```

---

## 12. Tailwind 开发技巧

### 重要提示

```html
<!-- 用 ! 前缀标记 !important -->
<div class="!bg-red-500">Always red</div>
```

### 选择器变体

```html
<!-- 父组件状态选择器 -->
<div class="group">
  <h3 class="group-hover:text-blue-500">Hover parent</h3>
</div>

<!-- 同级选择器 -->
<input class="peer" type="checkbox" />
<label class="peer-checked:text-blue-500">Checked label</label>

<!-- 子元素选择器 -->
<div class="[&>*]:border-b [&>*]:border-gray-200">
  <div>Each child has border</div>
  <div>Each child has border</div>
</div>
```

---

## 总结

- Tailwind 是 utility-first 的原子化 CSS 框架
- 移动优先，小屏幕样式为基础，大屏幕用前缀覆盖
- `@apply` 用于组合，但应适度使用
- 暗色模式用 `dark:` 前缀
- JIT 模式支持任意值
- 配合 `prettier-plugin-tailwindcss` 自动排序类名
- `class-variance-authority` 管理组件变体
