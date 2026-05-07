# Props 高级模式

## 1. children Prop

`children` 是 React 中一个特殊的 prop，代表组件标签之间的所有内容。

### 1.1 基本用法

```jsx
function Card({ children }) {
  return (
    <div className="card">
      {children}
    </div>
  );
}

// 使用
<Card>
  <h2>Title</h2>
  <p>Some content here</p>
  <button>Action</button>
</Card>
```

### 1.2 children 的类型

`children` 可以是任何可渲染的内容：

```jsx
// 字符串
<Alert>Something happened</Alert>

// 数字
<Display>{42}</Display>

// JSX 元素
<Container>
  <Child />
</Container>

// 数组
<List>
  <Item>One</Item>
  <Item>Two</Item>
  <Item>Three</Item>
</List>

// 函数（此时就是 Render Props 模式）
<DataProvider>
  {(data) => <Display data={data} />}
</DataProvider>

// null / undefined / boolean（不会渲染）
<Conditional>{showContent && <Content />}</Conditional>
```

### 1.3 组合式 children

```jsx
function Layout({ header, sidebar, footer, children }) {
  return (
    <div className="layout">
      <header>{header}</header>
      <div className="main">
        <aside>{sidebar}</aside>
        <main>{children}</main>
      </div>
      <footer>{footer}</footer>
    </div>
  );
}

// 使用
<Layout
  header={<NavBar />}
  sidebar={<SideMenu />}
  footer={<FooterBar />}
>
  <Article />
</Layout>
```

### 1.4 检查 children 类型

```jsx
import React from 'react';

function TabContainer({ children }) {
  // 验证 children 必须是 Tab 组件
  React.Children.forEach(children, (child) => {
    if (child.type !== Tab) {
      throw new Error('TabContainer only accepts Tab components');
    }
  });

  return <div className="tabs">{children}</div>;
}

// React.Children 工具方法
React.Children.map(children, fn)      // 遍历并映射
React.Children.forEach(children, fn)  // 遍历
React.Children.count(children)        // 计数
React.Children.only(children)         // 验证只有一个 child
React.Children.toArray(children)      // 转为数组
```

---

## 2. Render Props

Render Props 是一种通过 **将函数作为 prop 传入** 来共享代码的模式。传入的函数返回要渲染的内容。

### 2.1 基本概念

```jsx
function MouseTracker({ render }) {
  const [position, setPosition] = React.useState({ x: 0, y: 0 });

  const handleMouseMove = (e) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div onMouseMove={handleMouseMove}>
      {render(position)}  {/* 调用传入的 render 函数 */}
    </div>
  );
}

// 使用
<MouseTracker
  render={({ x, y }) => (
    <p>Mouse position: ({x}, {y})</p>
  )}
/>

<MouseTracker
  render={({ x, y }) => (
    <div
      className="cursor-dot"
      style={{ left: x, top: y }}
    />
  )}
/>
```

### 2.2 使用 children 作为 Render Prop

```jsx
function DataFetcher({ url, children }) {
  const [data, setData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    fetch(url)
      .then((res) => res.json())
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [url]);

  return children({ data, loading, error });
}

// 使用
<DataFetcher url="/api/users">
  {({ data, loading, error }) => {
    if (loading) return <Spinner />;
    if (error) return <ErrorMessage error={error} />;
    return <UserList users={data} />;
  }}
</DataFetcher>
```

### 2.3 与自定义 Hook 的对比

现代 React 中，许多 Render Props 场景可以用自定义 Hook 替代：

```jsx
// Render Props 方式
<MouseTracker render={({ x, y }) => <Dot x={x} y={y} />} />

// 自定义 Hook 方式（推荐）
function useMousePosition() {
  const [position, setPosition] = React.useState({ x: 0, y: 0 });

  React.useEffect(() => {
    const handler = (e) => setPosition({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', handler);
    return () => window.removeEventListener('mousemove', handler);
  }, []);

  return position;
}

function Dot() {
  const { x, y } = useMousePosition();
  return <div className="cursor-dot" style={{ left: x, top: y }} />;
}
```

> **何时使用 Render Props vs Hook**：如果逻辑与渲染紧密相关（如需要访问组件的渲染上下文），用 Render Props。如果逻辑独立于渲染，用自定义 Hook 更简洁。

---

## 3. 组件注入（Component Injection）

类似于 Render Props，但传递的是**组件本身**而非渲染函数。

```jsx
function List({ items, renderItem }) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={index}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}

// 传入渲染逻辑
<List
  items={users}
  renderItem={(user) => (
    <div>
      <strong>{user.name}</strong> - {user.email}
    </div>
  )}
/>

// 传入组件
<List
  items={products}
  renderItem={(product) => <ProductCard product={product} />}
/>
```

### 组件注入的另一种形式

直接传入组件类：

```jsx
function DataTable({ data, RowComponent }) {
  return (
    <table>
      <tbody>
        {data.map((item) => (
          <RowComponent key={item.id} data={item} />
        ))}
      </tbody>
    </table>
  );
}

// 使用
<DataTable
  data={users}
  RowComponent={UserRow}
/>
```

---

## 4. 多态组件（Polymorphic / "as" Prop）

通过 `as` prop 动态决定组件渲染的 HTML 元素或其他组件。

### 4.1 基本实现

```jsx
function Text({ as: Component = 'span', children, ...props }) {
  return <Component {...props}>{children}</Component>;
}

// 使用方式
<Text>Hello</Text>                          // 渲染为 <span>
<Text as="h1">Page Title</Text>             // 渲染为 <h1>
<Text as="p">A paragraph</Text>             // 渲染为 <p>
<Text as={Link} to="/about">About</Text>    // 渲染为 <Link>
```

### 4.2 完整示例

```jsx
function Button({ as: Component = 'button', variant = 'primary', size = 'medium', children, ...props }) {
  const className = `btn btn-${variant} btn-${size}`;

  return (
    <Component className={className} {...props}>
      {children}
    </Component>
  );
}

// 按钮
<Button onClick={handleClick}>Click me</Button>

// 链接外观的按钮
<Button as="a" href="/dashboard">Go to Dashboard</Button>

// 使用 React Router Link
<Button as={Link} to="/settings">Settings</Button>
```

### 4.3 TypeScript 中的多态组件

```tsx
// 完整的类型安全多态组件
type TextProps<C extends React.ElementType> = {
  as?: C;
  children: React.ReactNode;
} & React.ComponentPropsWithoutRef<C>;

function Text<C extends React.ElementType = 'span'>({
  as,
  children,
  ...props
}: TextProps<C>) {
  const Component = as || 'span';
  return <Component {...props}>{children}</Component>;
}

// 自动类型推断
<Text as="a" href="/about">About</Text>    // href 合法（HTMLAnchorElement 的属性）
<Text as="button" onClick={fn}>Click</Text> // onClick 合法
<Text as="h1" level={1}>Title</Text>        // level 不合法（非 h1 的属性）
```

---

## 5. Prop Getters

Prop Getters 是一种将 props 合并逻辑封装在函数中的模式，常见于自定义 Hook 库。

```jsx
function useToggle(initialState = false) {
  const [on, setOn] = React.useState(initialState);

  const toggle = () => setOn(!on);

  // 返回 prop getter
  const getToggleProps = (props = {}) => ({
    ...props,
    'aria-pressed': on,
    onClick: (e) => {
      props.onClick?.(e);
      toggle();
    },
  });

  return { on, toggle, getToggleProps };
}

// 使用
function ToggleButton() {
  const { on, getToggleProps } = useToggle();

  return (
    <button {...getToggleProps()}>
      {on ? 'ON' : 'OFF'}
    </button>
  );
}

// 用户可以传入额外的 props，不会被覆盖
<button {...getToggleProps({ className: 'my-btn', disabled: isDisabled })}>
  {on ? 'ON' : 'OFF'}
</button>
```

> Prop Getters 的价值在于：**合并用户传入的 props 与组件内部需要的 props**，避免冲突。

---

## 6. 复合组件模式（Compound Components）

复合组件是一组协同工作的组件，它们共享隐式的状态，无需显式传递 props。

### 6.1 基本示例：Tabs

```jsx
function Tabs({ children, defaultIndex = 0 }) {
  const [activeIndex, setActiveIndex] = React.useState(defaultIndex);

  // 通过 React.Children 或 Context 共享状态
  return (
    <TabsContext.Provider value={{ activeIndex, setActiveIndex }}>
      <div className="tabs">{children}</div>
    </TabsContext.Provider>
  );
}

function TabList({ children }) {
  return <div className="tab-list" role="tablist">{children}</div>;
}

function Tab({ index, children }) {
  const { activeIndex, setActiveIndex } = React.useContext(TabsContext);
  const isActive = index === activeIndex;

  return (
    <button
      role="tab"
      className={isActive ? 'tab active' : 'tab'}
      onClick={() => setActiveIndex(index)}
    >
      {children}
    </button>
  );
}

function TabPanel({ index, children }) {
  const { activeIndex } = React.useContext(TabsContext);

  if (index !== activeIndex) return null;
  return <div className="tab-panel">{children}</div>;
}

// 使用——优雅、声明式
<Tabs defaultIndex={0}>
  <TabList>
    <Tab index={0}>Tab 1</Tab>
    <Tab index={1}>Tab 2</Tab>
    <Tab index={2}>Tab 3</Tab>
  </TabList>
  <TabPanel index={0}>Content 1</TabPanel>
  <TabPanel index={1}>Content 2</TabPanel>
  <TabPanel index={2}>Content 3</TabPanel>
</Tabs>
```

### 6.2 使用 React.Children.map 自动注入

```jsx
function Tabs({ children, defaultIndex = 0 }) {
  const [activeIndex, setActiveIndex] = React.useState(defaultIndex);

  return React.Children.map(children, (child) => {
    // 将状态注入到每个 child
    return React.cloneElement(child, { activeIndex, setActiveIndex });
  });
}
```

### 6.3 著名的复合组件案例

- **Headless UI** 的 `<Menu>`, `<Listbox>`, `<Disclosure>`
- **Radix UI** 的各种复合组件
- **Chakra UI** 的 `<Menu>`, `<Tabs>` 等

```jsx
// Headless UI 示例
<Menu>
  <MenuButton>Options</MenuButton>
  <MenuItems>
    <MenuItem>
      {({ active }) => (
        <a className={active ? 'active' : ''} href="/account">
          Account settings
        </a>
      )}
    </MenuItem>
    <MenuItem>
      {({ active }) => (
        <a className={active ? 'active' : ''} href="/logout">
          Sign out
        </a>
      )}
    </MenuItem>
  </MenuItems>
</Menu>
```

---

## 7. 受控与非受控组件概念

虽然这两个术语最常用于表单，但其概念适用于所有组件。

### 7.1 受控组件（Controlled）

组件的值由父组件通过 props 控制，状态由外部管理。

```jsx
// 受控输入框
function ControlledInput({ value, onChange }) {
  return (
    <input
      value={value}           // 值来自 props
      onChange={(e) => onChange(e.target.value)}  // 变化通知父组件
    />
  );
}

// 使用
function App() {
  const [name, setName] = React.useState('');

  return <ControlledInput value={name} onChange={setName} />;
}
```

### 7.2 非受控组件（Uncontrolled）

组件自己管理内部状态，外部通过 ref 或初始值影响。

```jsx
// 非受控输入框
function UncontrolledInput({ defaultValue = '' }) {
  const [value, setValue] = React.useState(defaultValue);

  return (
    <input
      value={value}
      onChange={(e) => setValue(e.target.value)}
    />
  );
}
```

### 7.3 通用的受控/非受控模式

```jsx
// 同时支持受控和非受控
function Toggle({ controlledOn, defaultOn = false, onChange }) {
  // 如果传入了 controlledOn，使用受控模式
  const isControlled = controlledOn !== undefined;
  const [internalOn, setInternalOn] = React.useState(defaultOn);

  const on = isControlled ? controlledOn : internalOn;

  const handleToggle = () => {
    if (!isControlled) {
      setInternalOn(!on);
    }
    onChange?.(!on);
  };

  return (
    <button onClick={handleToggle}>
      {on ? 'ON' : 'OFF'}
    </button>
  );
}
```

---

## 8. 常见面试问题

### Q1：Render Props 和 HOC 有什么区别？

**Render Props** 通过函数传入渲染逻辑，更灵活，避免了 props 命名冲突和组件嵌套地狱。**HOC** 通过包装组件注入 props，可能导致来源不明的 props 和 Wrapper Hell。现代 React 更推荐自定义 Hook，它比两者都简洁。

### Q2：children 是不是一定要渲染？

不一定。可以检查 children 内容后决定是否渲染，也可以将 children 作为函数调用（Render Props 模式）。

### Q3：什么时候用 as prop？

当组件需要在不同场景下渲染为不同 HTML 元素时。比如 `Button` 可以渲染为 `<button>` 或 `<a>`。但不要过度使用——如果只在少数场景变化，不如创建独立的组件。
