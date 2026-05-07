# JSX 高级模式

## Render Props 模式

Render Props 是一种通过函数 prop 来决定组件渲染内容的模式：

```jsx
// 定义一个鼠标位置追踪组件
function MouseTracker({ render }) {
  const [position, setPosition] = useState({ x: 0, y: 0 })

  const handleMouseMove = (e) => {
    setPosition({ x: e.clientX, y: e.clientY })
  }

  return (
    <div onMouseMove={handleMouseMove} style={{ height: '300px', border: '1px solid #ccc' }}>
      {render(position)}
    </div>
  )
}

// 使用
function App() {
  return (
    <MouseTracker
      render={({ x, y }) => (
        <p>鼠标位置：({x}, {y})</p>
      )}
    />
  )
}
```

### 使用 children 作为 Render Props

```jsx
// 数据获取组件
function DataLoader({ url, children }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then((result) => {
        setData(result)
        setLoading(false)
      })
      .catch((err) => {
        setError(err)
        setLoading(false)
      })
  }, [url])

  // children 作为函数调用
  return children({ data, loading, error })
}

// 使用
function App() {
  return (
    <DataLoader url="/api/users">
      {({ data, loading, error }) => {
        if (loading) return <p>加载中...</p>
        if (error) return <p>出错了</p>
        return (
          <ul>
            {data.map((user) => (
              <li key={user.id}>{user.name}</li>
            ))}
          </ul>
        )
      }}
    </DataLoader>
  )
}
```

### 复用状态逻辑

```jsx
function Toggle({ children }) {
  const [isOn, setIsOn] = useState(false)

  const toggle = () => setIsOn((prev) => !prev)

  return children({ isOn, toggle })
}

// 同一个 Toggle 逻辑，不同 UI
function App() {
  return (
    <div>
      {/* 按钮形式 */}
      <Toggle>
        {({ isOn, toggle }) => (
          <button onClick={toggle}>
            {isOn ? '开' : '关'}
          </button>
        )}
      </Toggle>

      {/* 开关形式 */}
      <Toggle>
        {({ isOn, toggle }) => (
          <div
            onClick={toggle}
            style={{
              width: 50,
              height: 26,
              borderRadius: 13,
              backgroundColor: isOn ? '#1890ff' : '#d9d9d9',
              cursor: 'pointer',
            }}
          >
            <div
              style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                backgroundColor: 'white',
                transform: isOn ? 'translateX(26px)' : 'translateX(2px)',
                transition: 'transform 0.2s',
                marginTop: 2,
              }}
            />
          </div>
        )}
      </Toggle>
    </div>
  )
}
```

## 属性展开（Spread Attributes）

使用展开运算符将对象的所有属性一次性传递给组件：

```jsx
function Greeting({ name, age, city }) {
  return (
    <p>
      {name}，{age}岁，来自{city}
    </p>
  )
}

function App() {
  const user = { name: '张三', age: 25, city: '北京' }

  return (
    <div>
      {/* 手动传递 */}
      <Greeting name={user.name} age={user.age} city={user.city} />

      {/* 展开传递（效果相同） */}
      <Greeting {...user} />
    </div>
  )
}
```

### 选择性展开

```jsx
// 从 props 中提取部分属性，其余展开
function Input({ label, error, ...inputProps }) {
  return (
    <div className="form-field">
      <label>{label}</label>
      <input {...inputProps} className={error ? 'error' : ''} />
      {error && <span className="error-text">{error}</span>}
    </div>
  )
}

// 使用
<Input
  label="邮箱"
  type="email"
  name="email"
  placeholder="请输入邮箱"
  required
  error={errors.email}
/>
```

## 解构 Props

在组件函数参数中直接解构 props：

```jsx
// 基本解构
function UserCard({ name, email, avatar }) {
  return (
    <div>
      <img src={avatar} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
    </div>
  )
}

// 带默认值的解构
function Button({
  children,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  onClick,
}) {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  )
}

// 嵌套解构
function ProfileCard({
  user: { name, avatar },
  settings: { theme = 'light' },
}) {
  return (
    <div className={`card theme-${theme}`}>
      <img src={avatar} alt={name} />
      <h3>{name}</h3>
    </div>
  )
}

// 剩余参数
function FormField({ label, type = 'text', ...rest }) {
  return (
    <div>
      <label>{label}</label>
      <input type={type} {...rest} />
    </div>
  )
}
```

## dangerouslySetInnerHTML

当需要直接插入 HTML 字符串时使用，**有 XSS 攻击风险**：

```jsx
function Article({ htmlContent }) {
  return (
    <div
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  )
}
```

### XSS 风险警告

```jsx
// 危险：直接渲染用户输入的 HTML
function Comment({ userContent }) {
  // 如果 userContent 包含 <script>alert('xss')</script>
  // 将会导致 XSS 攻击
  return (
    <div dangerouslySetInnerHTML={{ __html: userContent }} />
  )
}
```

### 安全使用方式

```jsx
import DOMPurify from 'dompurify'

function SafeHtml({ content }) {
  // 使用 DOMPurify 清洗 HTML
  const sanitizedHtml = DOMPurify.sanitize(content)

  return (
    <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
  )
}
```

> 除非绝对必要（如渲染富文本编辑器的内容），否则不要使用 `dangerouslySetInnerHTML`。始终对用户输入的内容进行清洗。

## Portals（传送门）

Portal 将子节点渲染到父组件 DOM 层级之外的 DOM 节点中：

```jsx
import { createPortal } from 'react-dom'

function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null

  return createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          ✕
        </button>
        {children}
      </div>
    </div>,
    document.getElementById('modal-root'),
  )
}
```

```html
<!-- index.html -->
<body>
  <div id="root"></div>
  <div id="modal-root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
```

```jsx
// 使用
function App() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div>
      <button onClick={() => setIsOpen(true)}>打开弹窗</button>

      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)}>
        <h2>弹窗标题</h2>
        <p>弹窗内容</p>
      </Modal>
    </div>
  )
}
```

> Portal 的事件冒泡仍然遵循 React 组件树层级，而非 DOM 层级。

## 条件 Wrapper 组件

有时候需要根据条件决定是否添加包裹元素：

```jsx
function ConditionalWrapper({ condition, wrapper, children }) {
  return condition ? wrapper(children) : children
}

// 使用
function Card({ withBorder, withShadow, children }) {
  return (
    <ConditionalWrapper
      condition={withBorder}
      wrapper={(content) => <div className="border-wrapper">{content}</div>}
    >
      <ConditionalWrapper
        condition={withShadow}
        wrapper={(content) => <div className="shadow-wrapper">{content}</div>}
      >
        {children}
      </ConditionalWrapper>
    </ConditionalWrapper>
  )
}
```

### 更简洁的写法

```jsx
function Card({ withBorder, withShadow, children }) {
  const Wrapper = ({ children: c }) =>
    withBorder ? <div className="border">{c}</div> : c

  const ShadowWrapper = ({ children: c }) =>
    withShadow ? <div className="shadow">{c}</div> : c

  return (
    <Wrapper>
      <ShadowWrapper>
        {children}
      </ShadowWrapper>
    </Wrapper>
  )
}
```

## 高阶 JSX 模式

### 组合组件模式

```jsx
function Field({ label, error, children }) {
  return (
    <div className="field">
      <label className="field-label">{label}</label>
      {children}
      {error && <span className="field-error">{error}</span>}
    </div>
  )
}

// 使用组合
function LoginForm() {
  const [errors, setErrors] = useState({})

  return (
    <form>
      <Field label="用户名" error={errors.username}>
        <input type="text" name="username" />
      </Field>

      <Field label="密码" error={errors.password}>
        <input type="password" name="password" />
      </Field>

      <button type="submit">登录</button>
    </form>
  )
}
```

### 复合组件模式

```jsx
// 通过 Context 共享状态
const TabsContext = createContext()

function Tabs({ children, defaultIndex = 0 }) {
  const [activeIndex, setActiveIndex] = useState(defaultIndex)

  return (
    <TabsContext.Provider value={{ activeIndex, setActiveIndex }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  )
}

function TabList({ children }) {
  return <div className="tab-list">{children}</div>
}

function Tab({ index, children }) {
  const { activeIndex, setActiveIndex } = useContext(TabsContext)
  return (
    <button
      className={`tab ${activeIndex === index ? 'active' : ''}`}
      onClick={() => setActiveIndex(index)}
    >
      {children}
    </button>
  )
}

function TabPanel({ index, children }) {
  const { activeIndex } = useContext(TabsContext)
  if (activeIndex !== index) return null
  return <div className="tab-panel">{children}</div>
}

// 使用（注意组件嵌套和命名空间）
function App() {
  return (
    <Tabs defaultIndex={0}>
      <TabList>
        <Tab index={0}>标签一</Tab>
        <Tab index={1}>标签二</Tab>
        <Tab index={2}>标签三</Tab>
      </TabList>
      <TabPanel index={0}>内容一</TabPanel>
      <TabPanel index={1}>内容二</TabPanel>
      <TabPanel index={2}>内容三</TabPanel>
    </Tabs>
  )
}
```

## 小结

- **Render Props**：通过函数 prop 让调用者控制渲染内容，适合复用状态逻辑
- **属性展开**：简化 props 传递，结合解构实现选择性传递
- **解构 Props**：提高代码可读性，支持默认值和嵌套解构
- **dangerouslySetInnerHTML**：直接插入 HTML，必须对输入进行安全清洗
- **Portals**：将组件渲染到 DOM 树的其他位置，适合弹窗、通知等
- **条件 Wrapper**：根据条件动态添加包裹元素
- **复合组件**：通过 Context 共享状态，提供灵活的组合 API
