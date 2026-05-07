# Props 与 State 类型定义

## 1. Props 基础类型定义

### 必选与可选 Props

```tsx
interface ButtonProps {
  // 必选 props
  label: string
  onClick: () => void

  // 可选 props（使用 ? 标记）
  variant?: 'primary' | 'secondary' | 'danger'
  disabled?: boolean
  size?: 'sm' | 'md' | 'lg'
}

function Button({
  label,
  onClick,
  variant = 'primary',    // 带默认值的可选 prop
  disabled = false,
  size = 'md'
}: ButtonProps) {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  )
}
```

### 带默认值的三种方式

```tsx
// 方式一：解构默认值（推荐）
interface TextInputProps {
  value: string
  placeholder?: string
  maxLength?: number
}

function TextInput({
  value,
  placeholder = '请输入...',
  maxLength = 100
}: TextInputProps) {
  return <input value={value} placeholder={placeholder} maxLength={maxLength} />
}

// 方式二：defaultProps（不推荐，已过时）
TextInput.defaultProps = {
  placeholder: '请输入...',
  maxLength: 100
}

// 方式三：使用 Required<T> 使所有属性必选
type RequiredButtonProps = Required<ButtonProps>
// 此时所有属性都变为必选

// 使用 Partial<T> 使所有属性可选
type PartialButtonProps = Partial<ButtonProps>
// 此时所有属性都变为可选
```

---

## 2. children Prop 类型

```tsx
// React.ReactNode — 最通用，推荐
interface LayoutProps {
  children: React.ReactNode
}

function Layout({ children }: LayoutProps) {
  return <div className="layout">{children}</div>
}

// React.ReactElement — 只接受单个 React 元素
interface WrapperProps {
  children: React.ReactElement  // 必须是单个元素，不能是字符串或数组
}

function Wrapper({ children }: WrapperProps) {
  return <div className="wrapper">{children}</div>
}

// ReactElement<SpecificProps> — 只接受特定类型的元素
interface FieldProps {
  children: React.ReactElement<InputProps>  // 只接受 <Input /> 组件
}

function Field({ children }: FieldProps) {
  return <div className="field">{children}</div>
}

// 函数 children（Render Props 模式）
interface ListProps<T> {
  items: T[]
  children: (item: T, index: number) => React.ReactNode
}

function List<T>({ items, children }: ListProps<T>) {
  return <ul>{items.map((item, i) => children(item, i))}</ul>
}

// 使用
<List items={['apple', 'banana']}>
  {(item, index) => <li key={index}>{item}</li>}
</List>

// 多个 children（命名 slots 模式）
interface CardProps {
  header: React.ReactNode
  body: React.ReactNode
  footer?: React.ReactNode
}

function Card({ header, body, footer }: CardProps) {
  return (
    <div className="card">
      <div className="card-header">{header}</div>
      <div className="card-body">{body}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  )
}

// 或者使用复合组件模式
function CardHeader({ children }: { children: React.ReactNode }) {
  return <div className="card-header">{children}</div>
}
function CardBody({ children }: { children: React.ReactNode }) {
  return <div className="card-body">{children}</div>
}
```

---

## 3. 事件处理函数类型

### 常用事件类型

```tsx
// === 表单事件 ===

// 输入框变化
function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
  console.log(e.target.value)
}

// 文本域变化
function handleTextareaChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
  console.log(e.target.value)
}

// 下拉框变化
function handleSelectChange(e: React.ChangeEvent<HTMLSelectElement>) {
  console.log(e.target.value)
}

// 表单提交
function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
  e.preventDefault()
  const formData = new FormData(e.currentTarget)
}

// === 鼠标事件 ===

// 按钮点击
function handleClick(e: React.MouseEvent<HTMLButtonElement>) {
  console.log(e.currentTarget)    // HTMLButtonElement
  console.log(e.clientX, e.clientY) // 鼠标坐标
}

// div 点击
function handleDivClick(e: React.MouseEvent<HTMLDivElement>) {
  e.preventDefault()
}

// === 键盘事件 ===

function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
  if (e.key === 'Enter') {
    console.log('Enter pressed')
  }
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault()
    console.log('Ctrl+S pressed')
  }
}

// === 焦点事件 ===

function handleFocus(e: React.FocusEvent<HTMLInputElement>) {
  console.log('focused')
}

function handleBlur(e: React.FocusEvent<HTMLInputElement>) {
  console.log('blurred')
}
```

### 在 Props 中声明事件类型

```tsx
interface FormProps {
  // 方式一：使用 React.*Event 类型（推荐）
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void

  // 方式二：使用事件处理器类型别名
  onClick: React.MouseEventHandler<HTMLButtonElement>
  onKeyDown: React.KeyboardEventHandler<HTMLInputElement>

  // 方式三：自定义回调（不传 event，传业务数据）
  onSelect: (item: Item) => void
  onValueChange: (value: string) => void
}
```

### 事件类型速查表

| 事件 | 类型 | 常用目标元素 |
|------|------|-------------|
| onClick | `React.MouseEvent<T>` | button, div, a |
| onChange | `React.ChangeEvent<T>` | input, textarea, select |
| onSubmit | `React.FormEvent<T>` | form |
| onFocus/onBlur | `React.FocusEvent<T>` | input, button |
| onKeyDown/Up/Press | `React.KeyboardEvent<T>` | input, div |
| onMouseEnter/Leave | `React.MouseEvent<T>` | div, button |
| onScroll | `React.UIEvent<T>` | div |
| onDrag | `React.DragEvent<T>` | div |

> `T` 泛型参数为目标 DOM 元素类型。

---

## 4. 泛型组件

泛型组件允许组件在使用时确定具体类型：

```tsx
// 通用列表组件
interface ListProps<T> {
  items: T[]
  renderItem: (item: T, index: number) => React.ReactNode
  keyExtractor: (item: T) => string | number
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>
          {renderItem(item, index)}
        </li>
      ))}
    </ul>
  )
}

// 使用时自动推断 T 的类型
interface User {
  id: number
  name: string
}

const users: User[] = [
  { id: 1, name: 'Alice' },
  { id: 2, name: 'Bob' }
]

<List
  items={users}
  keyExtractor={(user) => user.id}       // user 类型自动推断为 User
  renderItem={(user) => <span>{user.name}</span>}
/>

// 通用选择器组件
interface SelectProps<T> {
  options: T[]
  value: T | null
  onChange: (value: T) => void
  getLabel: (item: T) => string
  getKey: (item: T) => string | number
}

function Select<T>({ options, value, onChange, getLabel, getKey }: SelectProps<T>) {
  return (
    <select
      value={value ? String(getKey(value)) : ''}
      onChange={(e) => {
        const item = options.find(o => String(getKey(o)) === e.target.value)
        if (item) onChange(item)
      }}
    >
      {options.map((option) => (
        <option key={getKey(option)} value={getKey(option)}>
          {getLabel(option)}
        </option>
      ))}
    </select>
  )
}

// 使用
<Select
  options={users}
  value={selectedUser}
  onChange={setSelectedUser}
  getLabel={(u) => u.name}
  getKey={(u) => u.id}
/>
```

### 带约束的泛型组件

```tsx
// T 必须包含 id 属性
interface DataTableProps<T extends { id: string | number }> {
  data: T[]
  columns: Array<{
    key: keyof T
    header: string
    render?: (value: T[keyof T], item: T) => React.ReactNode
  }>
}

function DataTable<T extends { id: string | number }>({
  data,
  columns
}: DataTableProps<T>) {
  return (
    <table>
      <thead>
        <tr>
          {columns.map(col => <th key={String(col.key)}>{col.header}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.map(item => (
          <tr key={item.id}>
            {columns.map(col => (
              <td key={String(col.key)}>
                {col.render
                  ? col.render(item[col.key], item)
                  : String(item[col.key])
                }
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

---

## 5. 判别联合类型（Discriminated Unions）

用于组件的多种变体，每种变体有不同 props：

```tsx
// 基本模式：共享 discriminant 字段
type AlertProps = {
  // 共享 props
  className?: string
  onClose?: () => void
} & (
  | {
      variant: 'success'
      message: string
    }
  | {
      variant: 'error'
      message: string
      errorCode: number
      retry?: () => void
    }
  | {
      variant: 'warning'
      message: string
      dismissAfter?: number
    }
)

function Alert(props: AlertProps) {
  // TypeScript 自动根据 variant 窄化类型
  switch (props.variant) {
    case 'success':
      // props 在这里是 SuccessAlert 类型
      return <div className="alert-success">{props.message}</div>

    case 'error':
      // props 在这里有 errorCode 和 retry
      return (
        <div className="alert-error">
          <span>{props.message}</span>
          <span>错误码: {props.errorCode}</span>
          {props.retry && <button onClick={props.retry}>重试</button>}
        </div>
      )

    case 'warning':
      return <div className="alert-warning">{props.message}</div>
  }
}

// 使用
<Alert variant="success" message="操作成功" />
<Alert variant="error" message="操作失败" errorCode={500} retry={() => {}} />
```

### 更复杂的例子：模态框变体

```tsx
type ModalProps = {
  isOpen: boolean
  onClose: () => void
} & (
  | {
      type: 'confirm'
      title: string
      message: string
      onConfirm: () => void
      confirmText?: string
      cancelText?: string
    }
  | {
      type: 'form'
      title: string
      onSubmit: (data: FormData) => void
      initialValues?: Record<string, string>
    }
  | {
      type: 'info'
      title: string
      content: React.ReactNode
    }
)

function Modal(props: ModalProps) {
  if (!props.isOpen) return null

  switch (props.type) {
    case 'confirm':
      return (
        <div className="modal">
          <h2>{props.title}</h2>
          <p>{props.message}</p>
          <button onClick={props.onConfirm}>
            {props.confirmText ?? '确认'}
          </button>
          <button onClick={props.onClose}>
            {props.cancelText ?? '取消'}
          </button>
        </div>
      )

    case 'form':
      return (
        <div className="modal">
          <h2>{props.title}</h2>
          <form onSubmit={(e) => {
            e.preventDefault()
            props.onSubmit(new FormData(e.currentTarget))
          }}>
            {/* 表单内容 */}
            <button type="submit">提交</button>
          </form>
        </div>
      )

    case 'info':
      return (
        <div className="modal">
          <h2>{props.title}</h2>
          {props.content}
          <button onClick={props.onClose}>关闭</button>
        </div>
      )
  }
}
```

---

## 6. 多态 "as" Prop 类型

让组件能够渲染为不同的 HTML 元素或组件：

```tsx
// 基础实现
type TextOwnProps<C extends React.ElementType = 'span'> = {
  as?: C
  children: React.ReactNode
  color?: 'primary' | 'secondary' | 'muted'
  size?: 'sm' | 'md' | 'lg'
}

// 合并组件自身的 props 和目标元素的 props
type TextProps<C extends React.ElementType = 'span'> =
  TextOwnProps<C> &
  Omit<React.ComponentPropsWithoutRef<C>, keyof TextOwnProps<C>>

function Text<C extends React.ElementType = 'span'>({
  as,
  children,
  color = 'primary',
  size = 'md',
  ...rest
}: TextProps<C>) {
  const Component = as || 'span'
  return (
    <Component
      className={`text-${color} text-${size}`}
      {...rest}
    >
      {children}
    </Component>
  )
}

// 使用
<Text>Hello</Text>                      // 渲染为 <span>
<Text as="h1">Page Title</Text>         // 渲染为 <h1>，支持 h1 的所有属性
<Text as="p" className="intro">...</Text> // 渲染为 <p>，支持 className
<Text as="a" href="/about">Link</Text>  // 渲染为 <a>，支持 href

// 如果 as="a" 但没传 href，TypeScript 会报错 ✅
```

### 完整的多态按钮组件

```tsx
type ButtonOwnProps<C extends React.ElementType = 'button'> = {
  as?: C
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  children: React.ReactNode
}

type ButtonProps<C extends React.ElementType = 'button'> =
  ButtonOwnProps<C> &
  Omit<React.ComponentPropsWithoutRef<C>, keyof ButtonOwnProps<C>>

function Button<C extends React.ElementType = 'button'>({
  as,
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  children,
  ...rest
}: ButtonProps<C>) {
  const Component = as || 'button'
  return (
    <Component
      className={`btn btn-${variant} btn-${size}`}
      disabled={loading}
      {...rest}
    >
      {loading && <Spinner />}
      {leftIcon && <span className="btn-icon">{leftIcon}</span>}
      {children}
      {rightIcon && <span className="btn-icon">{rightIcon}</span>}
    </Component>
  )
}

// 使用
<Button onClick={() => {}}>Click me</Button>
<Button as="a" href="/home">Go Home</Button>
<Button as={Link} to="/dashboard">Dashboard</Button>  // React Router Link
```

---

## 7. State 类型定义

### 基本 State 类型

```tsx
// 简单状态
function Counter() {
  const [count, setCount] = useState<number>(0)
  // TypeScript 自动推断，不需要显式类型
  const [name, setName] = useState('')  // 类型推断为 string
}

// 对象状态
interface FormState {
  username: string
  email: string
  age: number
  agreed: boolean
}

function Form() {
  const [form, setForm] = useState<FormState>({
    username: '',
    email: '',
    age: 0,
    agreed: false
  })

  // 部分更新
  const updateField = <K extends keyof FormState>(
    field: K,
    value: FormState[K]
  ) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }
}

// 可空状态
function UserProfile() {
  const [user, setUser] = useState<User | null>(null)
  const [error, setError] = useState<string | null>(null)
}
```

### 联合类型状态

```tsx
// 加载状态
type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: Error }

function UserList() {
  const [state, setState] = useState<AsyncState<User[]>>({ status: 'idle' })

  const fetchUsers = async () => {
    setState({ status: 'loading' })
    try {
      const res = await fetch('/api/users')
      const data = await res.json()
      setState({ status: 'success', data })
    } catch (err) {
      setState({ status: 'error', error: err as Error })
    }
  }

  switch (state.status) {
    case 'idle':
      return <button onClick={fetchUsers}>加载用户</button>
    case 'loading':
      return <Spinner />
    case 'success':
      return <ul>{state.data.map(u => <li key={u.id}>{u.name}</li>)}</ul>
    case 'error':
      return <div>错误: {state.error.message}</div>
  }
}
```

---

## 8. 高级 Props 模式

### Props 组合与工具类型

```tsx
// 使用 Pick 选择部分属性
type ButtonBaseProps = {
  label: string
  onClick: () => void
  disabled?: boolean
  variant?: 'primary' | 'secondary'
}

// 只需要 label 和 onClick
type SimpleButtonProps = Pick<ButtonBaseProps, 'label' | 'onClick'>

// 使用 Omit 排除属性
type ButtonWithoutVariant = Omit<ButtonBaseProps, 'variant'>

// 使用 Record 类型
interface TabProps {
  tabs: Record<string, React.ReactNode>  // { [key: string]: ReactNode }
}

// 使用 Extract / Exclude
type AllVariants = 'primary' | 'secondary' | 'danger' | 'ghost'
type ColorVariants = Extract<AllVariants, 'primary' | 'danger'>  // 'primary' | 'danger'
type NonColorVariants = Exclude<AllVariants, 'primary' | 'danger'> // 'secondary' | 'ghost'
```

### 条件 Props

```tsx
// 根据某个 prop 值决定其他 prop 是否必须
type ConditionalProps =
  | {
      hasIcon: true
      icon: React.ReactNode        // hasIcon 为 true 时必传
      iconPosition?: 'left' | 'right'
    }
  | {
      hasIcon: false
      icon?: never                   // hasIcon 为 false 时不能传
    }

type CompleteProps = ConditionalProps & {
  label: string
  onClick: () => void
}

function SmartButton(props: CompleteProps) {
  if (props.hasIcon) {
    // TypeScript 知道 props.icon 存在
    return <button>{props.icon}{props.label}</button>
  }
  return <button>{props.label}</button>
}

// 使用
<SmartButton hasIcon={true} icon={<Star />} label="Star" />
<SmartButton hasIcon={false} label="No Icon" />
// <SmartButton hasIcon={false} icon={<Star />} label="..." /> // ❌ 编译错误
```

### 模板字面量类型用于变体

```tsx
// 使用模板字面量生成所有变体组合
type Size = 'sm' | 'md' | 'lg'
type Color = 'primary' | 'secondary' | 'danger'
type Variant = `${Size}-${Color}`
// 'sm-primary' | 'sm-secondary' | 'sm-danger'
// | 'md-primary' | 'md-secondary' | 'md-danger'
// | 'lg-primary' | 'lg-secondary' | 'lg-danger'

interface BadgeProps {
  variant: Variant
  children: React.ReactNode
}

function Badge({ variant, children }: BadgeProps) {
  return <span className={`badge badge-${variant}`}>{children}</span>
}

<Badge variant="sm-primary">Small</Badge>
<Badge variant="lg-danger">Large Danger</Badge>
```

---

## 总结

- Props 使用 `interface` 或 `type` 定义，必选用 `?` 标记可选
- children 用 `React.ReactNode`，Render Props 用函数类型
- 事件类型用 `React.XxxEvent<ElementType>`
- 复杂变体组件用**判别联合类型**
- 多态组件用**泛型 + `as` prop** 模式
- State 使用联合类型可以实现精确的状态管理
