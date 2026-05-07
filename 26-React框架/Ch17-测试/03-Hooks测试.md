# Hooks 测试

## 一、renderHook 基础

### 1.1 为什么需要 renderHook

```tsx
// Hooks 只能在函数组件中调用，无法直接测试
// ❌ 错误：直接调用 hook
function testUseCounter() {
  const { count, increment } = useCounter();  // Error!
}

// ❌ 错误：创建组件测试（过于繁琐）
function TestComponent() {
  const { count, increment } = useCounter();
  return <div>{count}</div>;
}

// ✅ 正确：使用 renderHook
import { renderHook } from '@testing-library/react';
const { result } = renderHook(() => useCounter());
```

### 1.2 renderHook 返回值

```tsx
const {
  result,         // { current: hook 返回值 }
  rerender,       // 重新渲染（可传新 props）
  unmount,        // 卸载组件
  waitForNextUpdate,  // 等待下一次更新（已废弃，用 waitFor）
} = renderHook(
  ({ initialValue }) => useCounter(initialValue),  // hook 回调
  {
    initialProps: { initialValue: 0 },  // 初始 props
    wrapper: ProviderWrapper,           // Context 包装
  }
);
```

---

## 二、act() 处理状态更新

### 2.1 为什么需要 act()

```tsx
import { renderHook, act } from '@testing-library/react';

// ❌ 不用 act 会报 warning
const { result } = renderHook(() => useCounter());
result.current.increment();  // Warning: update not wrapped in act()

// ✅ 使用 act 包裹状态更新
act(() => {
  result.current.increment();
});
expect(result.current.count).toBe(1);
```

**act() 的作用：** 确保所有 React 更新（state、effect、DOM）在断言前完成。

### 2.2 常见的 act() 使用场景

```tsx
// 1. 同步状态更新
act(() => {
  result.current.setCount(5);
});
expect(result.current.count).toBe(5);

// 2. 调用返回函数
act(() => {
  result.current.increment();
});
act(() => {
  result.current.decrement();
});

// 3. 异步状态更新需要 await
await act(async () => {
  await result.current.fetchData();
});

// 4. 定时器
vi.useFakeTimers();
act(() => {
  vi.advanceTimersByTime(1000);
});
vi.useRealTimers();
```

---

## 三、测试自定义 Hooks

### 3.1 测试 useState 逻辑

```tsx
// useToggle.ts
export function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);
  const toggle = useCallback(() => setValue(v => !v), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);
  return { value, toggle, setTrue, setFalse } as const;
}

// useToggle.test.ts
import { renderHook, act } from '@testing-library/react';
import { useToggle } from './useToggle';

describe('useToggle', () => {
  it('should default to false', () => {
    const { result } = renderHook(() => useToggle());
    expect(result.current.value).toBe(false);
  });

  it('should accept initial value', () => {
    const { result } = renderHook(() => useToggle(true));
    expect(result.current.value).toBe(true);
  });

  it('should toggle value', () => {
    const { result } = renderHook(() => useToggle(false));

    act(() => {
      result.current.toggle();
    });
    expect(result.current.value).toBe(true);

    act(() => {
      result.current.toggle();
    });
    expect(result.current.value).toBe(false);
  });

  it('should set true', () => {
    const { result } = renderHook(() => useToggle(false));

    act(() => {
      result.current.setTrue();
    });
    expect(result.current.value).toBe(true);
  });

  it('should set false', () => {
    const { result } = renderHook(() => useToggle(true));

    act(() => {
      result.current.setFalse();
    });
    expect(result.current.value).toBe(false);
  });

  // 测试函数引用稳定性
  it('should maintain stable function references', () => {
    const { result, rerender } = renderHook(() => useToggle());

    const { toggle, setTrue, setFalse } = result.current;

    rerender();

    expect(result.current.toggle).toBe(toggle);
    expect(result.current.setTrue).toBe(setTrue);
    expect(result.current.setFalse).toBe(setFalse);
  });
});
```

### 3.2 测试 useEffect 和副作用

```tsx
// useDocumentTitle.ts
export function useDocumentTitle(title: string) {
  useEffect(() => {
    const originalTitle = document.title;
    document.title = title;

    return () => {
      document.title = originalTitle;  // 清理函数
    };
  }, [title]);
}

// useDocumentTitle.test.ts
describe('useDocumentTitle', () => {
  it('should set document title', () => {
    renderHook(() => useDocumentTitle('新标题'));
    expect(document.title).toBe('新标题');
  });

  it('should update title when argument changes', () => {
    const { rerender } = renderHook(
      ({ title }) => useDocumentTitle(title),
      { initialProps: { title: '标题1' } }
    );

    expect(document.title).toBe('标题1');

    rerender({ title: '标题2' });
    expect(document.title).toBe('标题2');
  });

  it('should restore original title on unmount', () => {
    const originalTitle = document.title;

    const { unmount } = renderHook(() => useDocumentTitle('临时标题'));
    expect(document.title).toBe('临时标题');

    unmount();
    expect(document.title).toBe(originalTitle);
  });
});
```

### 3.3 测试 useMemo/useCallback 引用稳定性

```tsx
// useStableCallback.ts
export function useStableCallback<T extends (...args: any[]) => any>(callback: T): T {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useCallback(((...args) => callbackRef.current(...args)) as T, []);
}

// 测试
describe('useStableCallback', () => {
  it('should return a stable function reference', () => {
    const { result, rerender } = renderHook(
      ({ fn }) => useStableCallback(fn),
      { initialProps: { fn: vi.fn() } }
    );

    const firstRef = result.current;

    // 用新函数重新渲染
    rerender({ fn: vi.fn() });

    // 引用应该保持稳定
    expect(result.current).toBe(firstRef);
  });

  it('should call the latest callback', () => {
    const fn1 = vi.fn().mockReturnValue('v1');
    const fn2 = vi.fn().mockReturnValue('v2');

    const { result, rerender } = renderHook(
      ({ fn }) => useStableCallback(fn),
      { initialProps: { fn: fn1 } }
    );

    const stableFn = result.current;

    // 调用稳定函数 → 应调用 fn1
    expect(stableFn()).toBe('v1');

    // 更新 callback
    rerender({ fn: fn2 });

    // 再次调用稳定函数 → 应调用 fn2（最新闭包）
    expect(stableFn()).toBe('v2');
  });
});
```

---

## 四、测试异步 Hooks

### 4.1 测试 fetch 类 Hook

```tsx
// useFetch.ts
export function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(json => {
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      })
      .catch(err => {
        if (!cancelled) setError(err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [url]);

  return { data, loading, error };
}

// useFetch.test.ts
import { renderHook, waitFor } from '@testing-library/react';

// Mock global fetch
vi.stubGlobal('fetch', vi.fn());

describe('useFetch', () => {
  beforeEach(() => {
    vi.mocked(fetch).mockClear();
  });

  it('should start with loading state', () => {
    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));  // never resolves

    const { result } = renderHook(() => useFetch('/api/users'));

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('should fetch data successfully', async () => {
    const mockData = [{ id: 1, name: '张三' }];
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    } as Response);

    const { result } = renderHook(() => useFetch('/api/users'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toEqual(mockData);
    expect(result.current.error).toBeNull();
  });

  it('should handle fetch error', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 404,
    } as Response);

    const { result } = renderHook(() => useFetch('/api/users'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe('HTTP 404');
  });

  it('should handle network error', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network Error'));

    const { result } = renderHook(() => useFetch('/api/users'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error?.message).toBe('Network Error');
  });

  it('should refetch when url changes', async () => {
    const mockData1 = [{ id: 1 }];
    const mockData2 = [{ id: 2 }];

    vi.mocked(fetch)
      .mockResolvedValueOnce({ ok: true, json: async () => mockData1 } as Response)
      .mockResolvedValueOnce({ ok: true, json: async () => mockData2 } as Response);

    const { result, rerender } = renderHook(
      ({ url }) => useFetch(url),
      { initialProps: { url: '/api/users' } }
    );

    await waitFor(() => {
      expect(result.current.data).toEqual(mockData1);
    });

    // 改变 URL
    rerender({ url: '/api/posts' });

    await waitFor(() => {
      expect(result.current.data).toEqual(mockData2);
    });

    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('should cancel request on unmount', async () => {
    const abortSpy = vi.fn();
    const mockAbortController = {
      abort: abortSpy,
      signal: {} as AbortSignal,
    };
    vi.stubGlobal('AbortController', vi.fn(() => mockAbortController));

    vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));

    const { unmount } = renderHook(() => useFetch('/api/users'));

    unmount();

    // 注意：需要在 useFetch 中使用 AbortController 才能验证
  });
});
```

### 4.2 测试 useDebounce

```tsx
// useDebounce.test.ts
import { renderHook, act } from '@testing-library/react';
import { useDebounce } from './useDebounce';

describe('useDebounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should return initial value immediately', () => {
    const { result } = renderHook(() => useDebounce('hello', 500));
    expect(result.current).toBe('hello');
  });

  it('should debounce value updates', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'initial', delay: 500 } }
    );

    expect(result.current).toBe('initial');

    // 更新值
    rerender({ value: 'updated', delay: 500 });

    // 在延迟时间内，值未变
    expect(result.current).toBe('initial');

    // 快进 500ms
    act(() => {
      vi.advanceTimersByTime(500);
    });

    // 现在值更新了
    expect(result.current).toBe('updated');
  });

  it('should reset timer on rapid changes', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounce(value, 500),
      { initialProps: { value: 'a' } }
    );

    rerender({ value: 'b' });
    act(() => { vi.advanceTimersByTime(300); });  // 300ms，未到 500

    rerender({ value: 'c' });
    act(() => { vi.advanceTimersByTime(300); });  // 又 300ms，但计时器重置了

    expect(result.current).toBe('a');  // 仍未更新

    act(() => { vi.advanceTimersByTime(200); });  // 再 200ms = 到达 500ms
    expect(result.current).toBe('c');  // 最终值是 'c'
  });
});
```

---

## 五、测试依赖 Context 的 Hooks

### 5.1 使用 wrapper 提供 Context

```tsx
// useAuth.ts
export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}

// useAuth.test.ts
import { renderHook, act } from '@testing-library/react';
import { AuthProvider } from './AuthContext';
import { useAuth } from './useAuth';

describe('useAuth', () => {
  // 创建 wrapper
  const wrapper = ({ children }) => (
    <AuthProvider>{children}</AuthProvider>
  );

  it('should provide initial unauthenticated state', () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('should login successfully', async () => {
    vi.mocked(authApi.login).mockResolvedValue({
      user: { id: 1, name: '张三' },
      token: 'abc123',
    });

    const { result } = renderHook(() => useAuth(), { wrapper });

    await act(async () => {
      await result.current.login('zhangsan@example.com', 'password123');
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.name).toBe('张三');
  });

  it('should logout', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });

    // 先登录
    await act(async () => {
      await result.current.login('zhangsan@example.com', 'password123');
    });
    expect(result.current.isAuthenticated).toBe(true);

    // 登出
    act(() => {
      result.current.logout();
    });
    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
  });

  it('should throw when used outside provider', () => {
    // 不传 wrapper
    expect(() => {
      renderHook(() => useAuth());
    }).toThrow('useAuth must be used within AuthProvider');
  });
});
```

### 5.2 多 Context 组合

```tsx
// 测试依赖多个 Context 的 hook
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return function Wrapper({ children }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <ThemeProvider>
            {children}
          </ThemeProvider>
        </AuthProvider>
      </QueryClientProvider>
    );
  };
}

describe('useUserProfile', () => {
  const wrapper = createWrapper();

  it('should fetch profile for authenticated user', async () => {
    vi.mocked(api.getProfile).mockResolvedValue({ name: '张三' });

    const { result } = renderHook(() => useUserProfile(), { wrapper });

    await waitFor(() => {
      expect(result.current.data?.name).toBe('张三');
    });
  });
});
```

---

## 六、Hooks 测试中的 Mock

### 6.1 Mock 外部依赖

```tsx
// useGeolocation.ts 依赖 navigator.geolocation
// 测试时需要 mock

describe('useGeolocation', () => {
  it('should return coordinates when permission granted', async () => {
    const mockPosition = {
      coords: { latitude: 39.9, longitude: 116.4 },
    };

    // Mock navigator.geolocation
    vi.stubGlobal('navigator', {
      ...navigator,
      geolocation: {
        getCurrentPosition: vi.fn().mockImplementation((success) => {
          success(mockPosition);
        }),
      },
    });

    const { result } = renderHook(() => useGeolocation());

    await waitFor(() => {
      expect(result.current.latitude).toBe(39.9);
      expect(result.current.longitude).toBe(116.4);
    });
  });

  it('should handle permission denied', async () => {
    vi.stubGlobal('navigator', {
      ...navigator,
      geolocation: {
        getCurrentPosition: vi.fn().mockImplementation((_, error) => {
          error({ code: 1, message: 'User denied' });
        }),
      },
    });

    const { result } = renderHook(() => useGeolocation());

    await waitFor(() => {
      expect(result.current.error).toBe('User denied');
    });
  });
});
```

### 6.2 Mock 其他 Hooks

```tsx
// 测试 useUserProfile 时 mock useAuth
vi.mock('./useAuth', () => ({
  useAuth: vi.fn().mockReturnValue({
    isAuthenticated: true,
    user: { id: 1, token: 'mock-token' },
    login: vi.fn(),
    logout: vi.fn(),
  }),
}));

it('should fetch profile when authenticated', async () => {
  const { result } = renderHook(() => useUserProfile());
  // useAuth 被 mock 为已认证状态
  await waitFor(() => {
    expect(result.current.profile).toBeDefined();
  });
});
```

---

## 七、集成测试 vs 单元测试 Hooks

### 7.1 单元测试（单独测试 hook）

```tsx
// 适合：纯逻辑 hooks（useState, useReducer, useMemo 等）
describe('useCounter - 单元测试', () => {
  it('increments count', () => {
    const { result } = renderHook(() => useCounter());
    act(() => { result.current.increment(); });
    expect(result.current.count).toBe(1);
  });
});
```

### 7.2 集成测试（通过组件测试 hook）

```tsx
// 适合：涉及 DOM、事件、Context 的 hooks
function CounterTest() {
  const { count, increment, decrement } = useCounter();
  return (
    <div>
      <span>Count: {count}</span>
      <button onClick={increment}>+1</button>
      <button onClick={decrement}>-1</button>
    </div>
  );
}

describe('useCounter - 集成测试', () => {
  it('increments via button click', async () => {
    const user = userEvent.setup();
    render(<CounterTest />);

    await user.click(screen.getByRole('button', { name: '+1' }));
    expect(screen.getByText('Count: 1')).toBeInTheDocument();
  });
});
```

### 7.3 如何选择

```
纯逻辑 hook（计算、转换）    → 单元测试 renderHook
涉及 DOM/事件的 hook        → 集成测试（渲染组件）
涉及 Context 的 hook        → 集成测试 或 renderHook + wrapper
异步 hook（fetch、定时器）   → renderHook + waitFor
```

---

## 八、实战示例：useLocalStorage

```tsx
// useLocalStorage.ts
export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    setStoredValue(prev => {
      const valueToStore = value instanceof Function ? value(prev) : value;
      try {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      } catch (error) {
        console.error(`Error saving to localStorage key "${key}":`, error);
      }
      return valueToStore;
    });
  }, [key]);

  const removeValue = useCallback(() => {
    try {
      window.localStorage.removeItem(key);
      setStoredValue(initialValue);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  }, [key, initialValue]);

  return [storedValue, setValue, removeValue] as const;
}

// useLocalStorage.test.ts
import { renderHook, act } from '@testing-library/react';
import { useLocalStorage } from './useLocalStorage';

describe('useLocalStorage', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should return initial value when localStorage is empty', () => {
    const { result } = renderHook(() => useLocalStorage('key', 'default'));
    expect(result.current[0]).toBe('default');
  });

  it('should return stored value from localStorage', () => {
    localStorage.setItem('key', JSON.stringify('stored'));
    const { result } = renderHook(() => useLocalStorage('key', 'default'));
    expect(result.current[0]).toBe('stored');
  });

  it('should update localStorage when value changes', () => {
    const { result } = renderHook(() => useLocalStorage('key', 'initial'));

    act(() => {
      result.current[1]('updated');
    });

    expect(result.current[0]).toBe('updated');
    expect(JSON.parse(localStorage.getItem('key')!)).toBe('updated');
  });

  it('should support functional updates', () => {
    localStorage.setItem('count', JSON.stringify(0));
    const { result } = renderHook(() => useLocalStorage('count', 0));

    act(() => {
      result.current[1](prev => prev + 1);
    });

    expect(result.current[0]).toBe(1);
  });

  it('should remove value from localStorage', () => {
    localStorage.setItem('key', JSON.stringify('value'));
    const { result } = renderHook(() => useLocalStorage('key', 'default'));

    act(() => {
      result.current[2]();  // removeValue
    });

    expect(result.current[0]).toBe('default');
    expect(localStorage.getItem('key')).toBeNull();
  });

  it('should handle complex objects', () => {
    const initial = { name: '张三', settings: { theme: 'dark' } };
    const { result } = renderHook(() => useLocalStorage('user', initial));

    act(() => {
      result.current[1]({ ...result.current[0], name: '李四' });
    });

    expect(result.current[0].name).toBe('李四');
    expect(result.current[0].settings.theme).toBe('dark');
  });

  it('should handle corrupted localStorage data', () => {
    localStorage.setItem('key', 'not-valid-json');
    const { result } = renderHook(() => useLocalStorage('key', 'fallback'));
    expect(result.current[0]).toBe('fallback');
  });

  it('should handle localStorage errors', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('QuotaExceededError');
    });

    const { result } = renderHook(() => useLocalStorage('key', 'initial'));

    act(() => {
      result.current[1]('new value');
    });

    expect(console.error).toHaveBeenCalled();
  });
});
```

---

## 总结

| 概念 | 要点 |
|------|------|
| **renderHook** | 测试 hook 的标准方式，提供 result、rerender、unmount |
| **act()** | 包裹所有状态更新操作，确保 React 完成更新 |
| **wrapper** | 通过 wrapper 选项提供 Context |
| **异步 hook** | 使用 waitFor 等待异步状态变化 |
| **fake timers** | 配合 vi.useFakeTimers() 测试定时器相关 hook |
| **引用稳定性** | 测试 useCallback/useMemo 确保函数/值的引用不变 |
| **集成 vs 单元** | 纯逻辑用 renderHook，涉及 DOM 用组件测试 |
