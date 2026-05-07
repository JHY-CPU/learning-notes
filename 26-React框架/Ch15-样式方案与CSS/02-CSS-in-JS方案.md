# CSS-in-JS 方案

## 1. CSS-in-JS 概述

CSS-in-JS 是将 CSS 样式写在 JavaScript/TypeScript 中的方案，样式与组件绑定，实现真正的组件级隔离。

### 主流方案对比

| 方案 | 特点 | 运行时 | React 18 支持 |
|------|------|--------|--------------|
| **styled-components v6** | Tagged Template Literal | 有 | 支持 |
| **Emotion** | 支持多种写法 | 有 | 支持 |
| **Vanilla Extract** | 零运行时 | 无 | 支持 |
| **Panda CSS** | 零运行时，原子化 | 无 | 支持 |
| **Linaria** | 零运行时 | 无 | 支持 |

---

## 2. styled-components v6

### 安装与基础用法

```bash
npm install styled-components
npm install -D @types/styled-components  # TypeScript 项目
```

```tsx
import styled from 'styled-components'

// 创建样式化组件
const Container = styled.div`
  display: flex;
  flex-direction: column;
  padding: 16px;
  border-radius: 8px;
  background: white;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`

const Title = styled.h2`
  font-size: 24px;
  color: #333;
  margin-bottom: 8px;
`

const Button = styled.button`
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background: #007bff;
  color: white;
  cursor: pointer;
  transition: background 0.2s;

  &:hover {
    background: #0056b3;
  }

  &:disabled {
    background: #ccc;
    cursor: not-allowed;
  }
`

// 使用
function Card() {
  return (
    <Container>
      <Title>Card Title</Title>
      <Button>Click Me</Button>
    </Container>
  )
}
```

### 动态 Props

```tsx
import styled, { css } from 'styled-components'

interface ButtonProps {
  $variant?: 'primary' | 'secondary' | 'danger'
  $size?: 'sm' | 'md' | 'lg'
  $fullWidth?: boolean
}

// 使用 transient props（$ 前缀）避免传递到 DOM
const StyledButton = styled.button<ButtonProps>`
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;

  /* Size 变体 */
  ${({ $size = 'md' }) => {
    switch ($size) {
      case 'sm': return css`padding: 4px 8px; font-size: 12px;`
      case 'md': return css`padding: 8px 16px; font-size: 14px;`
      case 'lg': return css`padding: 12px 24px; font-size: 16px;`
    }
  }}

  /* Color 变体 */
  ${({ $variant = 'primary' }) => {
    switch ($variant) {
      case 'primary': return css`
        background: #007bff;
        color: white;
        &:hover { background: #0056b3; }
      `
      case 'secondary': return css`
        background: #6c757d;
        color: white;
        &:hover { background: #545b62; }
      `
      case 'danger': return css`
        background: #dc3545;
        color: white;
        &:hover { background: #c82333; }
      `
    }
  }}

  /* 宽度 */
  ${({ $fullWidth }) => $fullWidth && css`
    width: 100%;
  `}
`

// 使用
function App() {
  return (
    <div>
      <StyledButton $variant="primary" $size="lg">Primary</StyledButton>
      <StyledButton $variant="danger" $size="sm" $fullWidth>Delete</StyledButton>
    </div>
  )
}
```

> **Transient Props（$ 前缀）**：styled-components v5.1+ 支持 `$` 前缀，表示这个 prop 只用于样式计算，不会传递到真实 DOM 元素上。

### 组件扩展

```tsx
// 方式一：extends
const IconButton = styled(StyledButton)`
  display: inline-flex;
  align-items: center;
  gap: 8px;

  svg {
    width: 16px;
    height: 16px;
  }
`

// 方式二：attrs — 设置默认属性
const Input = styled.input.attrs({
  type: 'text',
  autoComplete: 'off',
  placeholder: 'Enter text...'
})`
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;

  &:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }
`

// 方式三：基于 props 动态选择基础组件
const Text = styled.span<{ as?: 'p' | 'h1' | 'h2' | 'h3' }>`
  color: #333;
`
// 使用时
<Text as="h1">Title</Text>
<Text as="p">Paragraph</Text>
```

---

## 3. Theme Provider

### 配置主题

```tsx
// theme.ts
export const lightTheme = {
  colors: {
    primary: '#007bff',
    secondary: '#6c757d',
    danger: '#dc3545',
    success: '#28a745',
    background: '#ffffff',
    surface: '#f8f9fa',
    text: '#333333',
    textSecondary: '#6c757d',
    border: '#e0e0e0',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '16px',
    full: '9999px',
  },
  shadows: {
    sm: '0 1px 2px rgba(0,0,0,0.05)',
    md: '0 4px 6px rgba(0,0,0,0.1)',
    lg: '0 10px 15px rgba(0,0,0,0.1)',
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: {
      xs: '12px',
      sm: '14px',
      md: '16px',
      lg: '20px',
      xl: '24px',
    },
  }
} as const

export const darkTheme: typeof lightTheme = {
  colors: {
    primary: '#4dabf7',
    secondary: '#adb5bd',
    danger: '#ff6b6b',
    success: '#51cf66',
    background: '#1a1a2e',
    surface: '#16213e',
    text: '#e0e0e0',
    textSecondary: '#adb5bd',
    border: '#404040',
  },
  spacing: lightTheme.spacing,
  borderRadius: lightTheme.borderRadius,
  shadows: {
    sm: '0 1px 2px rgba(0,0,0,0.3)',
    md: '0 4px 6px rgba(0,0,0,0.4)',
    lg: '0 10px 15px rgba(0,0,0,0.4)',
  },
  typography: lightTheme.typography,
}

export type Theme = typeof lightTheme
```

### TypeScript 类型扩展

```ts
// styled.d.ts
import 'styled-components'

declare module 'styled-components' {
  export interface DefaultTheme {
    colors: {
      primary: string
      secondary: string
      danger: string
      success: string
      background: string
      surface: string
      text: string
      textSecondary: string
      border: string
    }
    spacing: {
      xs: string
      sm: string
      md: string
      lg: string
      xl: string
    }
    borderRadius: {
      sm: string
      md: string
      lg: string
      full: string
    }
    shadows: {
      sm: string
      md: string
      lg: string
    }
    typography: {
      fontFamily: string
      fontSize: {
        xs: string
        sm: string
        md: string
        lg: string
        xl: string
      }
    }
  }
}
```

### 使用主题

```tsx
// App.tsx
import { ThemeProvider } from 'styled-components'
import { lightTheme, darkTheme } from './theme'

function App() {
  const [isDark, setIsDark] = useState(false)

  return (
    <ThemeProvider theme={isDark ? darkTheme : lightTheme}>
      <GlobalStyle />
      <MainContent />
    </ThemeProvider>
  )
}

// 在组件中使用主题
const Card = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  padding: ${({ theme }) => theme.spacing.md};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  box-shadow: ${({ theme }) => theme.shadows.sm};
  border: 1px solid ${({ theme }) => theme.colors.border};
`
```

---

## 4. 全局样式

```tsx
import { createGlobalStyle } from 'styled-components'

const GlobalStyle = createGlobalStyle`
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: ${({ theme }) => theme.typography.fontFamily};
    background-color: ${({ theme }) => theme.colors.background};
    color: ${({ theme }) => theme.colors.text};
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    transition: background-color 0.3s, color 0.3s;
  }

  a {
    color: ${({ theme }) => theme.colors.primary};
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }
`

// 在 App 中使用
function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      {children}
    </ThemeProvider>
  )
}
```

---

## 5. Emotion

Emotion 是另一个主流 CSS-in-JS 库，API 更灵活。

### 安装

```bash
npm install @emotion/react @emotion/styled
```

### 两种用法

```tsx
// 方式一：css prop（需要配置 JSX 运行时）
/** @jsxImportSource @emotion/react */
import { css } from '@emotion/react'

const style = css`
  padding: 16px;
  background: #f0f0f0;
  border-radius: 8px;
`

function Card() {
  return <div css={style}>Content</div>
}

// 方式二：styled API（与 styled-components 类似）
import styled from '@emotion/styled'

const Container = styled.div`
  display: flex;
  padding: 16px;
`

const Button = styled.button<{ variant: 'primary' | 'secondary' }>`
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background: ${props => props.variant === 'primary' ? '#007bff' : '#6c757d'};
  color: white;
`
```

### Emotion vs styled-components

| 对比 | Emotion | styled-components |
|------|---------|-------------------|
| API | 多种（css prop, styled, css``） | 主要使用 styled |
| 包大小 | 更小（模块化） | 稍大 |
| 性能 | 略快 | 接近 |
| 主题化 | ThemeProvider | ThemeProvider |
| SSR | 支持 | 支持 |

---

## 6. 服务端渲染（SSR）

### styled-components SSR

```tsx
// server.tsx — Next.js / 自定义 SSR
import { ServerStyleSheet } from 'styled-components'

function renderToString(App: React.ReactElement) {
  const sheet = new ServerStyleSheet()

  try {
    const html = renderToString(sheet.collectStyles(App))
    const styleTags = sheet.getStyleTags()

    return { html, styleTags }
  } finally {
    sheet.seal()
  }
}

// 在 HTML 中注入
function Html({ html, styleTags }: { html: string; styleTags: string }) {
  return (
    <html>
      <head dangerouslySetInnerHTML={{ __html: styleTags }} />
      <body dangerouslySetInnerHTML={{ __html: html }} />
    </html>
  )
}
```

### Emotion SSR

```tsx
import createEmotionServer from '@emotion/server/create-instance'
import createCache from '@emotion/cache'

const cache = createCache({ key: 'css' })
const { extractCriticalToChunks, constructStyleTagsFromChunks } = createEmotionServer(cache)
```

---

## 7. 性能优化

### 避免内联创建样式

```tsx
// ❌ 每次渲染都创建新组件
function App() {
  const Button = styled.button`color: ${props => props.color};`
  return <Button color="red">Click</Button>
}

// ✅ 在组件外部定义
const Button = styled.button<{ color: string }>`color: ${props => props.color};`
function App() {
  return <Button color="red">Click</Button>
}
```

### 使用 shouldComponentUpdate / memo

```tsx
import styled from 'styled-components'

// 使用 .withConfig 避免不必要的重新渲染
const OptimizedButton = styled.button.withConfig({
  shouldForwardProp: (prop) => !['$variant', '$loading'].includes(prop),
})`
  padding: 8px 16px;
  /* ... */
`
```

### CSS 变量结合

```tsx
// 对于频繁变化的样式，使用 CSS 变量避免重新生成样式
const Box = styled.div<{ $color: string }>`
  /* 使用 CSS 变量传递动态值 */
  --box-color: ${({ $color }) => $color};
  background: var(--box-color);
  border: 2px solid var(--box-color);
`
```

---

## 8. 零运行时方案

### Vanilla Extract（推荐用于生产）

```bash
npm install @vanilla-extract/css @vanilla-extract/vite-plugin
```

```tsx
// button.css.ts
import { style, recipe } from '@vanilla-extract/css'
import { vars } from './theme.css'

export const buttonRecipe = recipe({
  base: {
    padding: '8px 16px',
    borderRadius: '4px',
    cursor: 'pointer',
    border: 'none',
  },
  variants: {
    variant: {
      primary: {
        background: vars.colors.primary,
        color: 'white',
      },
      secondary: {
        background: vars.colors.secondary,
        color: 'white',
      },
    },
    size: {
      sm: { padding: '4px 8px', fontSize: '12px' },
      md: { padding: '8px 16px', fontSize: '14px' },
      lg: { padding: '12px 24px', fontSize: '16px' },
    },
  },
  defaultVariants: {
    variant: 'primary',
    size: 'md',
  },
})
```

```tsx
// Button.tsx — 零运行时
import { buttonRecipe } from './button.css'

interface ButtonProps {
  variant?: 'primary' | 'secondary'
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
}

export function Button({ variant, size, children }: ButtonProps) {
  return (
    <button className={buttonRecipe({ variant, size })}>
      {children}
    </button>
  )
}
```

---

## 9. 实用技巧

### TypeScript 类型辅助

```tsx
// 从组件推断 props 类型
import { ComponentProps } from 'react'

const StyledButton = styled.button<{ $variant: 'primary' | 'secondary' }>`...`

// 获取样式化组件的 props
type ButtonProps = ComponentProps<typeof StyledButton>

// 扩展 props
type ExtendedButtonProps = ButtonProps & {
  loading?: boolean
  icon?: React.ReactNode
}
```

### keyframes 动画

```tsx
import styled, { keyframes } from 'styled-components'

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`

const Spinner = styled.div`
  width: 24px;
  height: 24px;
  border: 3px solid #e0e0e0;
  border-top-color: #007bff;
  border-radius: 50%;
  animation: ${spin} 0.8s linear infinite;
`

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
`

const FadeIn = styled.div`
  animation: ${fadeIn} 0.3s ease-out;
`
```

### 响应式断点

```tsx
const breakpoints = {
  sm: '576px',
  md: '768px',
  lg: '992px',
  xl: '1200px',
}

const media = {
  sm: `@media (min-width: ${breakpoints.sm})`,
  md: `@media (min-width: ${breakpoints.md})`,
  lg: `@media (min-width: ${breakpoints.lg})`,
  xl: `@media (min-width: ${breakpoints.xl})`,
}

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;

  ${media.sm} {
    grid-template-columns: repeat(2, 1fr);
  }

  ${media.md} {
    grid-template-columns: repeat(3, 1fr);
  }

  ${media.lg} {
    grid-template-columns: repeat(4, 1fr);
  }
`
```

---

## 总结

- **styled-components** 和 **Emotion** 是主流 CSS-in-JS 方案，API 相似
- 使用 **transient props（$ 前缀）** 避免 props 泄露到 DOM
- **ThemeProvider** 实现主题切换，配合 TypeScript 获得完整类型支持
- SSR 需要使用 `ServerStyleSheet` 收集样式
- **零运行时方案**（Vanilla Extract、Linaria）适合性能敏感项目
- 避免在渲染函数内创建样式组件
