# JSX 样式处理

## 内联样式（Inline Styles）

内联样式使用 JavaScript 对象传递，属性名使用驼峰命名：

```jsx
function HighlightText() {
  const styles = {
    color: '#1890ff',
    fontSize: '18px',
    fontWeight: 'bold',
    backgroundColor: '#f0f5ff',
    padding: '12px 16px',
    borderRadius: '4px',
    borderLeft: '4px solid #1890ff',
  }

  return <div style={styles}>这是一段高亮文字</div>
}
```

### 内联样式的特点

```jsx
function Card({ isHighlighted }) {
  return (
    <div
      style={{
        // 数字自动加 px 后缀（非以下属性除外）
        width: 200,             // → width: 200px
        height: 100,            // → height: 100px
        padding: 10,            // → padding: 10px
        // 属性名使用驼峰
        backgroundColor: isHighlighted ? '#fff7e6' : '#ffffff',
        border: '1px solid #d9d9d9',
        borderRadius: 8,
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        // margin 可简写
        marginTop: 16,
        marginBottom: 16,
      }}
    >
      <p style={{ color: '#333', lineHeight: 1.6 }}>卡片内容</p>
    </div>
  )
}
```

> 内联样式适合动态样式或少量样式，不适合伪类（`:hover`）、媒体查询等。

## CSS 文件导入

直接导入 `.css` 文件是最简单的方式：

```css
/* App.css */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.title {
  font-size: 24px;
  color: #333;
  margin-bottom: 16px;
}
```

```jsx
import './App.css'

function App() {
  return (
    <div className="container">
      <h1 className="title">标题</h1>
    </div>
  )
}
```

> 全局 CSS 的缺点：类名可能冲突，样式作用于整个应用。

## CSS Modules

CSS Modules 为每个 CSS 文件生成唯一的类名，避免命名冲突：

```css
/* Button.module.css */
.primary {
  background-color: #1890ff;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}

.primary:hover {
  background-color: #40a9ff;
}

.secondary {
  background-color: white;
  color: #1890ff;
  border: 1px solid #1890ff;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}

.secondary:hover {
  background-color: #f0f5ff;
}
```

```jsx
import styles from './Button.module.css'

function Button({ variant = 'primary', children, onClick }) {
  return (
    <button
      className={styles[variant]}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

// 使用
<Button variant="primary">主要按钮</Button>
<Button variant="secondary">次要按钮</Button>
```

### 多个类名

```jsx
import styles from './Card.module.css'

function Card({ isActive, children }) {
  // 方式一：模板字符串拼接
  const className = `${styles.card} ${isActive ? styles.active : ''}`

  // 方式二：数组 join
  const className2 = [
    styles.card,
    isActive && styles.active,
  ].filter(Boolean).join(' ')

  return <div className={className}>{children}</div>
}
```

### CSS Modules 中使用全局样式

```css
/* Card.module.css */
.card {
  border: 1px solid #d9d9d9;
  border-radius: 8px;
}

/* :global() 包裹的类名不会被模块化 */
:global(.ant-btn) {
  font-size: 14px;
}
```

## className 工具库

使用 `clsx` 或 `classnames` 库简化条件类名拼接：

```bash
pnpm add clsx
# 或
pnpm add classnames
```

### clsx 用法

```jsx
import clsx from 'clsx'
import styles from './Button.module.css'

function Button({ variant, size, disabled, isLoading, children }) {
  return (
    <button
      className={clsx(
        styles.button,
        // 字符串
        variant === 'primary' && styles.primary,
        variant === 'secondary' && styles.secondary,
        variant === 'danger' && styles.danger,
        // 对象形式
        {
          [styles.small]: size === 'small',
          [styles.large]: size === 'large',
          [styles.disabled]: disabled,
          [styles.loading]: isLoading,
        },
      )}
      disabled={disabled}
    >
      {isLoading ? '加载中...' : children}
    </button>
  )
}
```

### classnames 用法

```jsx
import classNames from 'classnames'
import styles from './Card.module.css'

function Card({ type, isBordered, isShadow, children }) {
  return (
    <div
      className={classNames(styles.card, {
        [styles.bordered]: isBordered,
        [styles.shadow]: isShadow,
        [styles[type]]: !!type,
      })}
    >
      {children}
    </div>
  )
}
```

## 动态样式

根据组件的状态或 props 动态计算样式：

```jsx
function ProgressBar({ percent }) {
  const barColor = percent < 30
    ? '#ff4d4f'
    : percent < 70
      ? '#faad14'
      : '#52c41a'

  return (
    <div style={{ backgroundColor: '#f0f0f0', borderRadius: 4, height: 8 }}>
      <div
        style={{
          width: `${percent}%`,
          backgroundColor: barColor,
          height: '100%',
          borderRadius: 4,
          transition: 'width 0.3s ease',
        }}
      />
    </div>
  )
}
```

### CSS 变量动态样式

```jsx
function ThemedButton({ color, size, children }) {
  return (
    <button
      style={{
        '--btn-color': color || '#1890ff',
        '--btn-size': size === 'large' ? '16px' : '14px',
      }}
      className="themed-btn"
    >
      {children}
    </button>
  )
}
```

```css
/* global.css */
.themed-btn {
  background-color: var(--btn-color);
  font-size: var(--btn-size);
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
}
```

## SCSS/SASS

Vite 天然支持 SCSS，只需安装预处理器：

```bash
pnpm add -D sass
```

```scss
// Card.module.scss
$primary: #1890ff;
$border-radius: 8px;

.card {
  border: 1px solid #d9d9d9;
  border-radius: $border-radius;
  padding: 16px;

  &:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  }

  .title {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
  }

  .content {
    color: #666;
    line-height: 1.6;
  }
}
```

```jsx
import styles from './Card.module.scss'

function Card({ title, content }) {
  return (
    <div className={styles.card}>
      <h3 className={styles.title}>{title}</h3>
      <p className={styles.content}>{content}</p>
    </div>
  )
}
```

## CSS-in-JS 简介

CSS-in-JS 将样式直接写在 JavaScript 中，提供更强大的动态样式能力：

```jsx
// styled-components 示例（需要安装）
// pnpm add styled-components

import styled from 'styled-components'

const Button = styled.button`
  background-color: ${(props) => (props.$primary ? '#1890ff' : 'white')};
  color: ${(props) => (props.$primary ? 'white' : '#1890ff')};
  border: 1px solid #1890ff;
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;

  &:hover {
    opacity: 0.8;
  }
`

function App() {
  return (
    <div>
      <Button $primary>主要按钮</Button>
      <Button>次要按钮</Button>
    </div>
  )
}
```

> 注意：CSS-in-JS 方案在运行时有一定性能开销，React 19 推荐使用 CSS Modules 或 Tailwind CSS。

## 样式方案对比

| 方案 | 作用域 | 动态样式 | 性能 | 推荐度 |
|------|--------|----------|------|--------|
| 内联样式 | 组件级 | 好 | 中 | 少量动态样式 |
| 全局 CSS | 全局 | 差 | 好 | 全局基础样式 |
| CSS Modules | 文件级 | 一般 | 好 | **推荐** |
| SCSS | 文件级 | 一般 | 好 | 复杂样式 |
| CSS-in-JS | 组件级 | 好 | 中 | 动态主题 |
| Tailwind | 原子级 | 好 | 好 | 快速开发 |

## 小结

- **内联样式**：适合动态样式，属性名用驼峰，数值自动加 px
- **CSS Modules**：推荐方案，类名自动隔离，避免冲突
- **clsx/classnames**：简化条件类名拼接
- **SCSS**：增强 CSS 能力（变量、嵌套、混合等）
- **CSS 变量**：CSS 和 JS 之间的桥梁，适合主题切换
- 根据项目需求选择合适的样式方案，小项目可用 CSS Modules，大项目考虑统一方案
