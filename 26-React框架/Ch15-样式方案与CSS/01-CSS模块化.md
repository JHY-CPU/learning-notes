# CSS 模块化

## 1. CSS Modules 基础

CSS Modules 是一种将 CSS 限定在组件范围内的方案，解决了全局 CSS 的命名冲突问题。

### 基本用法

```css
/* Button.module.css */
.container {
  display: inline-flex;
  align-items: center;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  border: none;
  font-size: 14px;
  transition: all 0.2s ease;
}

.primary {
  background-color: #007bff;
  color: white;
}

.primary:hover {
  background-color: #0056b3;
}

.secondary {
  background-color: #6c757d;
  color: white;
}

.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

```tsx
// Button.tsx
import styles from './Button.module.css'

interface ButtonProps {
  variant?: 'primary' | 'secondary'
  disabled?: boolean
  children: React.ReactNode
  onClick?: () => void
}

export function Button({ variant = 'primary', disabled, children, onClick }: ButtonProps) {
  return (
    <button
      className={`${styles.container} ${styles[variant]} ${disabled ? styles.disabled : ''}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  )
}
```

### 编译后的类名

CSS Modules 会将类名编译为唯一的哈希值，避免冲突：

```css
/* 编译前 */
.container { display: flex; }

/* 编译后 */
.Button_container_a3xK2 { display: flex; }
```

---

## 2. 样式组合（composes）

### 在 CSS 中组合样式

```css
/* styles.module.css */
.base {
  font-size: 14px;
  padding: 8px 16px;
  border-radius: 4px;
}

/* 使用 composes 组合其他类 */
.primaryButton {
  composes: base;          /* 继承 base 的样式 */
  background-color: #007bff;
  color: white;
}

.secondaryButton {
  composes: base;
  background-color: #6c757d;
  color: white;
}

/* 从其他文件组合 */
.anotherStyle {
  composes: someClass from './other.module.css';
}
```

```tsx
// 只需要引用一个类
<button className={styles.primaryButton}>Primary</button>
// 编译后的 class 包含 base 和 primaryButton 的所有样式
```

### composes vs 选择器嵌套

```css
/* ❌ 不推荐：用嵌套实现组合 */
.primary {
  font-size: 14px;
  padding: 8px;
}
.primary .icon {
  color: white;
}

/* ✅ 推荐：用 composes 实现组合 */
.base {
  font-size: 14px;
  padding: 8px;
}
.iconBase {
  margin-right: 8px;
}
.primary {
  composes: base;
  background: blue;
}
.primaryIcon {
  composes: iconBase;
  color: white;
}
```

---

## 3. 全局样式

CSS Modules 中默认所有类名都是局部作用域的。如果需要全局类名，使用 `:global` 语法：

```css
/* styles.module.css */

/* 局部类名（默认） */
.container {
  padding: 16px;
}

/* 全局类名 — 不会被编译为哈希值 */
:global(.global-button) {
  border: 1px solid #ccc;
}

/* 混合使用 */
.container :global(.ant-btn) {
  margin-left: 8px;
}

/* 全局选择器包裹 */
:global {
  .theme-dark .container {
    background: #1a1a1a;
    color: white;
  }
}
```

### 全局 CSS 文件

对于真正的全局样式（reset、字体等），使用普通 CSS 文件：

```css
/* src/index.css — 不要使用 .module 后缀 */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
  color: #333;
}

html {
  scroll-behavior: smooth;
}
```

```tsx
// main.tsx
import './index.css'  // 全局导入
import App from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(<App />)
```

---

## 4. TypeScript 类型声明

### 方式一：Vite 内置支持

Vite 自动支持 CSS Modules 的类型推断。如果使用 `vite-env.d.ts`：

```ts
/// <reference types="vite/client" />

// 这个声明已包含在 vite/client 中
// declare module '*.module.css' {
//   const classes: { readonly [key: string]: string }
//   export default classes
// }
```

### 方式二：手写类型声明

```ts
// src/types/css.d.ts
declare module '*.module.css' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.scss' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.sass' {
  const classes: { readonly [key: string]: string }
  export default classes
}

declare module '*.module.less' {
  const classes: { readonly [key: string]: string }
  export default classes
}
```

### 方式三：typed-css-modules（精确类型）

安装工具获取精确的类型提示：

```bash
npm install -D typed-css-modules
# 或
npm install -D @teamsupercell/typings-for-css-modules-loader
```

配置后会为每个 `.module.css` 文件生成对应的 `.d.ts` 文件：

```ts
// Button.module.css.d.ts（自动生成）
declare const styles: {
  readonly "container": string
  readonly "primary": string
  readonly "secondary": string
  readonly "disabled": string
}
export default styles
```

这样就能在 IDE 中获得精确的类名补全，拼错类名会立即报错。

---

## 5. CSS 变量与模块化主题

### 使用 CSS 自定义属性（CSS Variables）

```css
/* theme.module.css */
:root {
  --color-primary: #007bff;
  --color-secondary: #6c757d;
  --color-danger: #dc3545;
  --color-success: #28a745;
  --color-warning: #ffc107;

  --font-size-sm: 12px;
  --font-size-md: 14px;
  --font-size-lg: 16px;
  --font-size-xl: 20px;

  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --spacing-xl: 32px;

  --border-radius: 4px;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
}
```

```css
/* Button.module.css */
.button {
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: var(--font-size-md);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s;
}

.button:hover {
  box-shadow: var(--shadow-md);
}

.primary {
  background-color: var(--color-primary);
  color: white;
}

.danger {
  background-color: var(--color-danger);
  color: white;
}
```

### 通过 JavaScript 动态切换主题

```tsx
// ThemeProvider.tsx
import { createContext, useContext, useState, useEffect } from 'react'

type Theme = 'light' | 'dark'

const themeVariables: Record<Theme, Record<string, string>> = {
  light: {
    '--color-bg': '#ffffff',
    '--color-text': '#333333',
    '--color-border': '#e0e0e0',
  },
  dark: {
    '--color-bg': '#1a1a1a',
    '--color-text': '#e0e0e0',
    '--color-border': '#404040',
  }
}

function setCSSVariables(vars: Record<string, string>) {
  const root = document.documentElement
  Object.entries(vars).forEach(([key, value]) => {
    root.style.setProperty(key, value)
  })
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light')

  useEffect(() => {
    setCSSVariables(themeVariables[theme])
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

const ThemeContext = createContext<{
  theme: Theme
  setTheme: (t: Theme) => void
}>({ theme: 'light', setTheme: () => {} })

export const useTheme = () => useContext(ThemeContext)
```

---

## 6. 主题化方案

### 方案一：CSS 变量 + data 属性

```css
/* styles.module.css */
.container {
  background: var(--bg-primary);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

/* 定义不同主题的变量 */
[data-theme='light'] {
  --bg-primary: #ffffff;
  --text-primary: #333333;
  --border-color: #e0e0e0;
}

[data-theme='dark'] {
  --bg-primary: #1a1a1a;
  --text-primary: #e0e0e0;
  --border-color: #404040;
}
```

```tsx
// 切换主题
function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme')
  const next = current === 'dark' ? 'light' : 'dark'
  document.documentElement.setAttribute('data-theme', next)
}
```

### 方案二：多套 CSS 文件

```
styles/
  light.module.css    # 浅色主题变量
  dark.module.css     # 深色主题变量
```

```tsx
// 动态导入主题
const loadTheme = async (theme: 'light' | 'dark') => {
  await import(`./styles/${theme}.module.css`)
}
```

---

## 7. 组件样式组织模式

### 模式一：文件夹模式（推荐）

```
components/
  Button/
    Button.tsx
    Button.module.css
    Button.test.tsx
    Button.stories.tsx
    index.ts
```

```tsx
// Button/index.ts
export { Button } from './Button'
export type { ButtonProps } from './Button'
```

### 模式二：样式集中管理

```
src/
  styles/
    components/
      Button.module.css
      Card.module.css
    layout/
      Header.module.css
      Footer.module.css
    variables.css       # CSS 变量
    global.css          # 全局样式
  components/
    Button.tsx
    Card.tsx
```

---

## 8. CSS Modules 工具函数

```ts
// utils/cn.ts — 合并类名
export function cn(...args: (string | undefined | null | false | Record<string, boolean>)[]): string {
  return args
    .flatMap(arg => {
      if (!arg) return []
      if (typeof arg === 'string') return [arg]
      return Object.entries(arg)
        .filter(([, v]) => v)
        .map(([k]) => k)
    })
    .join(' ')
}

// 使用
import styles from './Button.module.css'

const className = cn(
  styles.container,
  variant === 'primary' && styles.primary,
  variant === 'secondary' && styles.secondary,
  disabled && styles.disabled,
  { [styles.loading]: isLoading }  // 条件类名
)

// 或使用 clsx / classnames 库
import clsx from 'clsx'

const className = clsx(
  styles.container,
  {
    [styles.primary]: variant === 'primary',
    [styles.secondary]: variant === 'secondary',
    [styles.disabled]: disabled,
  }
)
```

---

## 9. CSS Modules vs 其他方案对比

| 特性 | CSS Modules | 全局 CSS | CSS-in-JS | Tailwind |
|------|------------|---------|-----------|----------|
| 作用域隔离 | 类级隔离 | 无 | 组件级 | 无（靠约定） |
| 运行时开销 | 无 | 无 | 有 | 无 |
| TypeScript 支持 | 需配置 | 无 | 原生 | 需配置 |
| 动态样式 | 通过 CSS 变量 | 通过 CSS 变量 | 原生支持 | 有限 |
| 学习曲线 | 低 | 最低 | 中 | 中 |
| 打包大小 | 去重后小 | 可能大 | 运行时额外开销 | 只含使用到的 |

### 适用场景

- **CSS Modules**：组件化项目、需要零运行时开销、团队习惯纯 CSS
- **CSS-in-JS**：需要大量动态样式、喜欢 JS 写 CSS
- **Tailwind**：快速原型开发、设计系统统一

---

## 10. 常见问题

### 动态类名拼接

```tsx
// ❌ 动态拼接字符串（不安全）
<div className={`${styles[`card-${variant}`]}`}>...</div>

// ✅ 使用映射对象（安全）
const variantClass = {
  primary: styles.primary,
  secondary: styles.secondary,
  danger: styles.danger,
}[variant]

<div className={`${styles.container} ${variantClass}`}>...</div>
```

### 访问不存在的类名

```tsx
// ❌ 运行时返回 undefined，不会报错但样式丢失
<div className={styles.typo}>...</div>

// ✅ 使用 typed-css-modules 或手动检查
<div className={styles.container}>...</div>
```

### 与第三方组件库样式覆盖

```css
/* 覆盖第三方库样式 */
.container :global(.ant-btn-primary) {
  background-color: var(--color-primary);
}

/* 或使用更高的特异性 */
.wrapper.wrapper .button {
  /* 增加特异性 */
}
```

---

## 总结

- CSS Modules 通过编译时哈希类名实现样式隔离
- 使用 `composes` 在 CSS 层面组合样式
- 配合 CSS 变量实现主题切换
- 使用 `typed-css-modules` 获得精确的类型提示
- 推荐组件文件夹模式组织样式文件
