# 自定义 Hooks

## 目录

1. [为什么需要自定义 Hooks](#为什么需要自定义-hooks)
2. [命名规范与基本语法](#命名规范与基本语法)
3. [提取有状态逻辑](#提取有状态逻辑)
4. [实用自定义 Hooks 示例](#实用自定义-hooks-示例)
5. [Hooks 规则](#hooks-规则)
6. [Hooks 组合](#hooks-组合)

---

## 为什么需要自定义 Hooks

自定义 Hooks 让你将**组件逻辑**提取到可复用的函数中，实现逻辑共享而非 UI 复用。

### 复用逻辑 vs 复用 UI

```
传统复用方式（复用 UI）:
  - 高阶组件（HOC）→ 嵌套地狱、props 来源不明
  - Render Props → 回调嵌套、可读性差

自定义 Hooks（复用逻辑）:
  - 平铺组合、清晰明确
  - 每个 Hook 管理自己的状态
  - 没有组件嵌套
```

### 基本思想

```jsx
// 两个组件都需要"跟踪窗口大小"的逻辑
function ComponentA() {
  const [width, setWidth] = useState(window.innerWidth);
  const [height, setHeight] = useState(window.innerHeight);

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth);
      setHeight(window.innerHeight);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return <p>Component A: {width} x {height}</p>;
}

// ❌ 重复代码：ComponentB 需要同样的逻辑

// ✅ 提取为自定义 Hook
function useWindowSize() {
  const [size, setSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      setSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return size;
}

// 现在任何组件都可以复用
function ComponentA() {
  const { width, height } = useWindowSize();
  return <p>Component A: {width} x {height}</p>;
}

function ComponentB() {
  const { width } = useWindowSize();
  return <p>当前宽度: {width}px</p>;
}
```

---

## 命名规范与基本语法

### 命名规则

- 必须以 `use` 开头（`useXxx`）
- 这是 React 识别自定义 Hook 的约定，也是 ESLint 规则的要求

```jsx
// ✅ 正确的命名
useWindowSize()
useFetch()
useLocalStorage()
useDebounce()
useToggle()
useAuth()

// ❌ 错误的命名（React 不会将其视为 Hook）
fetchData()     // 没有 use 前缀
use_my_data()   // 不是 camelCase
```

### 基本结构

```jsx
function useCustomHook(initialValue) {
  // 1. 可以使用其他 Hooks
  const [state, setState] = useState(initialValue);
  const ref = useRef(null);

  // 2. 可以包含副作用
  useEffect(() => {
    // 副作用逻辑
    return () => {
      // 清理逻辑
    };
  }, []);

  // 3. 返回需要暴露的值和函数
  return [state, setState];
  // 或者返回对象
  // return { state, setState, ref };
}
```

---

## 提取有状态逻辑

### 示例：提取倒计时逻辑

```jsx
// ❌ 未提取：逻辑混在组件中
function VerificationCode({ phone }) {
  const [countdown, setCountdown] = useState(0);
  const [isDisabled, setIsDisabled] = useState(false);

  const sendCode = async () => {
    await api.sendVerificationCode(phone);
    setCountdown(60);
    setIsDisabled(true);
  };

  useEffect(() => {
    if (countdown <= 0) {
      setIsDisabled(false);
      return;
    }
    const timer = setInterval(() => {
      setCountdown(prev => prev - 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [countdown]);

  return (
    <button disabled={isDisabled} onClick={sendCode}>
      {countdown > 0 ? `${countdown} 秒后重发` : '发送验证码'}
    </button>
  );
}

// ✅ 提取为自定义 Hook
function useCountdown(seconds = 60) {
  const [countdown, setCountdown] = useState(0);
  const isRunning = countdown > 0;

  const start = useCallback(() => {
    setCountdown(seconds);
  }, [seconds]);

  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setInterval(() => {
      setCountdown(prev => prev - 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [countdown]);

  return { countdown, isRunning, start };
}

// 组件变得简洁
function VerificationCode({ phone }) {
  const { countdown, isRunning, start } = useCountdown(60);

  const sendCode = async () => {
    await api.sendVerificationCode(phone);
    start();
  };

  return (
    <button disabled={isRunning} onClick={sendCode}>
      {isRunning ? `${countdown} 秒后重发` : '发送验证码'}
    </button>
  );
}
```

---

## 实用自定义 Hooks 示例

### useLocalStorage

同步读写 localStorage，保持状态同步。

```jsx
function useLocalStorage(key, initialValue) {
  // 惰性初始化：只在首次渲染时读取 localStorage
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`读取 localStorage key "${key}" 失败:`, error);
      return initialValue;
    }
  });

  // 每次值变化时同步到 localStorage
  const setValue = useCallback((value) => {
    setStoredValue(prev => {
      const newValue = typeof value === 'function' ? value(prev) : value;
      try {
        window.localStorage.setItem(key, JSON.stringify(newValue));
      } catch (error) {
        console.warn(`写入 localStorage key "${key}" 失败:`, error);
      }
      return newValue;
    });
  }, [key]);

  // 监听其他标签页的修改
  useEffect(() => {
    const handleStorage = (e) => {
      if (e.key === key && e.newValue !== null) {
        setStoredValue(JSON.parse(e.newValue));
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, [key]);

  return [storedValue, setValue];
}

// 使用
function App() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  const [user, setUser] = useLocalStorage('user', { name: '', email: '' });

  return (
    <div className={`theme-${theme}`}>
      <button onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>
        切换主题
      </button>
      <input
        value={user.name}
        onChange={e => setUser(prev => ({ ...prev, name: e.target.value }))}
      />
    </div>
  );
}
```

### useFetch

通用的数据获取 Hook。

```jsx
function useFetch(url, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const controller = new AbortController();
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
}

// 使用
function UserProfile({ userId }) {
  const { data: user, loading, error } = useFetch(`/api/users/${userId}`);

  if (loading) return <Skeleton />;
  if (error) return <ErrorAlert message={error} />;

  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.email}</p>
    </div>
  );
}

function ProductList() {
  const { data: products, loading, refetch } = useFetch('/api/products');

  return (
    <div>
      <button onClick={refetch}>刷新</button>
      {loading ? <Spinner /> : products?.map(p => <ProductCard key={p.id} product={p} />)}
    </div>
  );
}
```

### useDebounce

防抖值：在值停止变化后延迟更新。

```jsx
function useDebounce(value, delay = 500) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(timer); // 值变化时清除上一个定时器
  }, [value, delay]);

  return debouncedValue;
}

// 使用：搜索防抖
function SearchPage() {
  const [keyword, setKeyword] = useState('');
  const debouncedKeyword = useDebounce(keyword, 300);

  // 只有用户停止输入 300ms 后才发起请求
  const { data: results, loading } = useFetch(
    debouncedKeyword ? `/api/search?q=${debouncedKeyword}` : null
  );

  return (
    <div>
      <input
        value={keyword}
        onChange={e => setKeyword(e.target.value)}
        placeholder="搜索..."
      />
      {loading && <p>搜索中...</p>}
      {results?.map(r => <ResultItem key={r.id} result={r} />)}
    </div>
  );
}
```

### useDebounceFn

防抖函数：延迟执行回调函数。

```jsx
function useDebounceFn(fn, delay = 500) {
  const timerRef = useRef(null);

  const debouncedFn = useCallback((...args) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      fn(...args);
    }, delay);
  }, [fn, delay]);

  // 组件卸载时清除定时器
  useEffect(() => {
    return () => clearTimeout(timerRef.current);
  }, []);

  // 返回取消函数
  const cancel = useCallback(() => {
    clearTimeout(timerRef.current);
  }, []);

  return { run: debouncedFn, cancel };
}

// 使用
function App() {
  const { run: debouncedSearch } = useDebounceFn((query) => {
    fetch(`/api/search?q=${query}`);
  }, 500);

  return <input onChange={e => debouncedSearch(e.target.value)} />;
}
```

### useToggle

布尔状态切换。

```jsx
function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);

  const toggle = useCallback(() => setValue(v => !v), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);

  return [value, { toggle, setTrue, setFalse, setValue }];
}

// 使用
function Modal() {
  const [isOpen, { toggle, setTrue, setFalse }] = useToggle(false);

  return (
    <div>
      <button onClick={setTrue}>打开弹窗</button>
      {isOpen && (
        <div className="modal">
          <h2>弹窗内容</h2>
          <button onClick={setFalse}>关闭</button>
        </div>
      )}
    </div>
  );
}
```

### useMediaQuery

响应式媒体查询。

```jsx
function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => {
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia(query);
    const handler = (e) => setMatches(e.matches);

    // 初始同步
    setMatches(mediaQuery.matches);

    // 监听变化
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, [query]);

  return matches;
}

// 使用
function App() {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isDark = useMediaQuery('(prefers-color-scheme: dark)');
  const prefersReducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)');

  return (
    <div>
      {isMobile ? <MobileNav /> : <DesktopNav />}
      <p>暗色模式: {isDark ? '是' : '否'}</p>
    </div>
  );
}
```

### useClickOutside

检测点击元素外部。

```jsx
function useClickOutside(ref, handler) {
  useEffect(() => {
    const listener = (event) => {
      // 点击的是 ref 内部，忽略
      if (!ref.current || ref.current.contains(event.target)) {
        return;
      }
      handler(event);
    };

    document.addEventListener('mousedown', listener);
    document.addEventListener('touchstart', listener);

    return () => {
      document.removeEventListener('mousedown', listener);
      document.removeEventListener('touchstart', listener);
    };
  }, [ref, handler]);
}

// 使用
function Dropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useClickOutside(dropdownRef, () => setIsOpen(false));

  return (
    <div ref={dropdownRef} className="dropdown">
      <button onClick={() => setIsOpen(!isOpen)}>菜单</button>
      {isOpen && (
        <ul className="dropdown-menu">
          <li>选项 1</li>
          <li>选项 2</li>
          <li>选项 3</li>
        </ul>
      )}
    </div>
  );
}
```

### useKeyPress

监听键盘按键。

```jsx
function useKeyPress(targetKey) {
  const [isPressed, setIsPressed] = useState(false);

  useEffect(() => {
    const handleDown = (e) => {
      if (e.key === targetKey) setIsPressed(true);
    };
    const handleUp = (e) => {
      if (e.key === targetKey) setIsPressed(false);
    };

    window.addEventListener('keydown', handleDown);
    window.addEventListener('keyup', handleUp);

    return () => {
      window.removeEventListener('keydown', handleDown);
      window.removeEventListener('keyup', handleUp);
    };
  }, [targetKey]);

  return isPressed;
}

// 使用
function App() {
  const isEnterPressed = useKeyPress('Enter');
  const isEscapePressed = useKeyPress('Escape');

  useEffect(() => {
    if (isEscapePressed) closeModal();
    if (isEnterPressed) submitForm();
  }, [isEnterPressed, isEscapePressed]);

  return <div>按 Enter 提交，按 Escape 关闭</div>;
}
```

### useIntersectionObserver

检测元素是否进入视口。

```jsx
function useIntersectionObserver(options = {}) {
  const [entry, setEntry] = useState(null);
  const [node, setNode] = useState(null);

  const observer = useRef(null);

  useEffect(() => {
    if (observer.current) observer.current.disconnect();

    observer.current = new IntersectionObserver(([entry]) => {
      setEntry(entry);
    }, options);

    if (node) observer.current.observe(node);

    return () => observer.current?.disconnect();
  }, [node, options.threshold, options.root, options.rootMargin]);

  return [setNode, entry];
}

// 使用：无限滚动
function InfiniteList({ loadMore }) {
  const [items, setItems] = useState([]);
  const [sentinelRef, entry] = useIntersectionObserver({ threshold: 0 });

  useEffect(() => {
    if (entry?.isIntersecting) {
      loadMore().then(newItems => {
        setItems(prev => [...prev, ...newItems]);
      });
    }
  }, [entry?.isIntersecting]);

  return (
    <ul>
      {items.map(item => <li key={item.id}>{item.name}</li>)}
      <li ref={sentinelRef}>加载更多...</li>
    </ul>
  );
}

// 使用：懒加载图片
function LazyImage({ src, alt }) {
  const [hasLoaded, setHasLoaded] = useState(false);
  const [ref, entry] = useIntersectionObserver({ threshold: 0 });

  useEffect(() => {
    if (entry?.isIntersecting && !hasLoaded) {
      const img = new Image();
      img.src = src;
      img.onload = () => setHasLoaded(true);
    }
  }, [entry?.isIntersecting, src]);

  return (
    <div ref={ref} className="lazy-image">
      {hasLoaded ? (
        <img src={src} alt={alt} />
      ) : (
        <div className="placeholder">加载中...</div>
      )}
    </div>
  );
}
```

---

## Hooks 规则

### 规则一：只在顶层调用 Hook

不要在循环、条件判断或嵌套函数中调用 Hook。

```jsx
function Component({ type }) {
  // ❌ 错误：在条件中调用 Hook
  if (type === 'A') {
    useEffect(() => { ... }, []);
  }

  // ❌ 错误：在循环中调用 Hook
  for (let i = 0; i < items.length; i++) {
    useState(items[i]);
  }

  // ✅ 正确：始终在顶层调用
  useEffect(() => {
    if (type === 'A') {
      // 把条件逻辑放在 Hook 内部
      doSomethingForA();
    }
  }, [type]);
}
```

### 规则二：只在函数组件和自定义 Hook 中调用

```jsx
// ✅ 函数组件
function MyComponent() {
  const [count, setCount] = useState(0);
  // ...
}

// ✅ 自定义 Hook
function useCount() {
  const [count, setCount] = useState(0);
  return { count, setCount };
}

// ❌ 普通 JavaScript 函数
function regularFunction() {
  useState(0); // 错误！
}

// ❌ 类组件
class MyClassComponent extends React.Component {
  render() {
    useState(0); // 错误！
  }
}
```

---

## Hooks 组合

自定义 Hooks 可以由其他 Hooks（包括其他自定义 Hooks）组合而成。

### 组合示例：useAuth

```jsx
function useAuth() {
  const [user, setUser] = useLocalStorage('auth-user', null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 验证 token
    const token = localStorage.getItem('auth-token');
    if (token) {
      api.verifyToken(token)
        .then(setUser)
        .catch(() => setUser(null))
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (credentials) => {
    const { user, token } = await api.login(credentials);
    localStorage.setItem('auth-token', token);
    setUser(user);
    return user;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('auth-token');
    setUser(null);
  }, []);

  return {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user,
  };
}

// 使用
function App() {
  const { user, loading, login, logout, isAuthenticated } = useAuth();

  if (loading) return <Spinner />;

  return isAuthenticated ? (
    <Dashboard user={user} onLogout={logout} />
  ) : (
    <LoginPage onLogin={login} />
  );
}
```

### 组合示例：useForm

```jsx
function useForm(initialValues, validate) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  const handleChange = useCallback((e) => {
    const { name, value, type, checked } = e.target;
    setValues(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  }, []);

  const handleBlur = useCallback((e) => {
    const { name } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
  }, []);

  // 验证
  useEffect(() => {
    if (validate) {
      const newErrors = validate(values);
      setErrors(newErrors);
    }
  }, [values]);

  const reset = useCallback(() => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  }, [initialValues]);

  // 为表单元素生成 props
  const getFieldProps = useCallback((name) => ({
    name,
    value: values[name] ?? '',
    onChange: handleChange,
    onBlur: handleBlur,
  }), [values, handleChange, handleBlur]);

  const isValid = Object.keys(errors).length === 0;

  return { values, errors, touched, isValid, handleChange, handleBlur, reset, getFieldProps };
}

// 使用
function LoginForm() {
  const { values, errors, isValid, getFieldProps, reset } = useForm(
    { email: '', password: '' },
    (values) => {
      const errors = {};
      if (!values.email) errors.email = '请输入邮箱';
      if (!values.password) errors.password = '请输入密码';
      return errors;
    }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isValid) {
      api.login(values);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <input {...getFieldProps('email')} type="email" placeholder="邮箱" />
        {errors.email && <span className="error">{errors.email}</span>}
      </div>
      <div>
        <input {...getFieldProps('password')} type="password" placeholder="密码" />
        {errors.password && <span className="error">{errors.password}</span>}
      </div>
      <button type="submit" disabled={!isValid}>登录</button>
    </form>
  );
}
```

---

## 总结

| 要点 | 说明 |
|------|------|
| 命名 | 必须以 `use` 开头 |
| 目的 | 复用**状态逻辑**，不是复用 UI |
| 组合 | 自定义 Hook 可以使用其他 Hooks |
| 隔离 | 每个调用都有独立的状态 |
| 规则 | 遵循 Hooks 的两条规则 |

自定义 Hooks 是 React 生态中最强大的代码复用方式之一。当你发现多个组件有相似的逻辑时，考虑将其提取为自定义 Hook。
