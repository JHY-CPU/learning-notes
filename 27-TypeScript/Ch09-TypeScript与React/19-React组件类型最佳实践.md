# React组件类型最佳实践

## 一、概念说明

良好的类型设计让 React 组件易于理解、使用和维护。最佳实践包括 Props 设计原则、泛型组件用法、类型导出策略等，这些实践能显著提升代码质量和开发体验。

## 二、具体用法

### 2.1 Props 设计原则

```tsx
// 好的 Props 设计 — 明确、可扩展、类型安全
interface DataTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  onRowClick?: (item: T) => void;
  emptyText?: string;
  loading?: boolean;
}

// ColumnDef 定义列配置
interface ColumnDef<T> {
  key: keyof T;
  header: string;
  render?: (value: T[keyof T], item: T) => React.ReactNode;
  width?: number | string;
}

// 泛型组件
function DataTable<T extends { id: number | string }>({
  data,
  columns,
  onRowClick,
  emptyText = '暂无数据',
  loading = false,
}: DataTableProps<T>) {
  if (loading) return <div>加载中...</div>;
  if (data.length === 0) return <div>{emptyText}</div>;

  return (
    <table>
      <thead>
        <tr>
          {columns.map(col => <th key={String(col.key)}>{col.header}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.map(item => (
          <tr key={item.id} onClick={() => onRowClick?.(item)}>
            {columns.map(col => (
              <td key={String(col.key)}>
                {col.render ? col.render(item[col.key], item) : String(item[col.key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

### 2.2 组件 Props 继承

```tsx
// 透传原生 HTML 属性
type ButtonProps = {
  variant?: 'primary' | 'secondary';
  isLoading?: boolean;
} & React.ComponentPropsWithoutRef<'button'>;

function Button({ variant = 'primary', isLoading, children, ...rest }: ButtonProps) {
  return (
    <button className={`btn-${variant}`} disabled={isLoading} {...rest}>
      {isLoading ? '加载中...' : children}
    </button>
  );
}

// 可以透传所有 button 属性
<Button type="submit" disabled variant="primary" onClick={() => {}}>提交</Button>;
```

### 2.3 Polymorphic 组件（多态组件）

```tsx
// 多态组件 — 可渲染为不同元素
type TextProps<C extends React.ElementType> = {
  as?: C;
  children: React.ReactNode;
  color?: 'primary' | 'secondary';
} & React.ComponentPropsWithoutRef<C>;

function Text<C extends React.ElementType = 'span'>({
  as,
  children,
  color,
  ...rest
}: TextProps<C>) {
  const Component = as || 'span';
  return (
    <Component className={`text-${color}`} {...rest}>
      {children}
    </Component>
  );
}

// 使用
<Text as="h1" color="primary">标题</Text>     // h1 的属性可用
<Text as="p" color="secondary">段落</Text>      // p 的属性可用
```

### 2.4 类型导出策略

```tsx
// 项目结构建议
// components/Button/
//   ├── Button.tsx        # 组件实现
//   ├── Button.test.tsx   # 测试
//   ├── index.ts          # 导出
//   └── types.ts          # 类型定义

// types.ts — 导出类型
export interface ButtonProps {
  variant: 'primary' | 'secondary';
  onClick: () => void;
  children: React.ReactNode;
}

// index.ts — 统一导出
export { Button } from './Button';
export type { ButtonProps } from './types';
```

### 2.5 组件默认值最佳实践

```tsx
// 使用解构默认值，而非 defaultProps
interface CardProps {
  title: string;
  padding?: number;      // 可选属性
  shadow?: boolean;      // 可选属性
  children: React.ReactNode;
}

function Card({
  title,
  padding = 16,          // 解构中设默认值
  shadow = true,
  children,
}: CardProps) {
  return (
    <div style={{ padding, boxShadow: shadow ? '0 2px 8px rgba(0,0,0,0.1)' : 'none' }}>
      <h2>{title}</h2>
      {children}
    </div>
  );
}
```

## 三、注意事项与常见陷阱

1. **Props 接口优先用 `interface`**：可被扩展，支持 `extends`
2. **不要过度使用泛型**：只有在组件需要处理多种数据类型时才用
3. **透传原生属性用 `ComponentPropsWithoutRef`**：避免手动重写所有 HTML 属性
4. **使用 `export type` 导出类型**：`isolatedModules` 模式下更安全
5. **`defaultProps` 已不推荐**：使用解构默认值代替
