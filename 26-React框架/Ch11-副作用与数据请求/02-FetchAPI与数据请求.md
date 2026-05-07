# Ch11-02 Fetch API 与数据请求

## 目录

1. [fetch() 基础](#1-fetch-基础)
2. [HTTP 方法：GET/POST/PUT/DELETE](#2-http-方法getpostputdelete)
3. [处理 JSON 数据](#3-处理-json-数据)
4. [错误处理](#4-错误处理)
5. [AbortController 取消请求](#5-abortcontroller-取消请求)
6. [Loading/Error/Data 状态模式](#6-loadingerrordata-状态模式)
7. [自定义 useFetch Hook](#7-自定义-usefetch-hook)
8. [请求去重 (Request Deduplication)](#8-请求去重-request-deduplication)

---

## 1. fetch() 基础

### 1.1 基本语法

```js
fetch(url, options)
  .then((response) => {
    // response 是 Response 对象
  })
  .catch((error) => {
    // 网络错误（DNS 解析失败、断网等）
  });
```

### 1.2 Response 对象的常用属性和方法

```js
const response = await fetch("https://api.example.com/data");

// 状态信息
response.status; // 200, 404, 500 等
response.statusText; // "OK", "Not Found"
response.ok; // true (200-299) 或 false
response.redirected; // 是否经过重定向
response.type; // "basic", "cors", "opaque"
response.url; // 最终的 URL（考虑重定向后）

// 读取响应体（只能调用一个！）
await response.text(); // 返回字符串
await response.json(); // 返回解析后的 JSON 对象
await response.blob(); // 返回 Blob 对象（二进制数据）
await response.arrayBuffer(); // 返回 ArrayBuffer
await response.formData(); // 返回 FormData

// 响应头
response.headers.get("Content-Type");
response.headers.forEach((value, key) => {
  console.log(`${key}: ${value}`);
});
```

### 1.3 最简单的请求

```jsx
import { useEffect, useState } from "react";

function UserList() {
  const [users, setUsers] = useState([]);

  useEffect(() => {
    fetch("https://jsonplaceholder.typicode.com/users")
      .then((res) => res.json())
      .then(setUsers);
  }, []);

  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

---

## 2. HTTP 方法：GET/POST/PUT/DELETE

### 2.1 GET 请求

```js
// 最简形式 - GET 是默认方法
const response = await fetch("/api/users");

// 带查询参数
const params = new URLSearchParams({
  page: "1",
  limit: "10",
  search: "john",
});
const response = await fetch(`/api/users?${params}`);

// 带自定义 headers
const response = await fetch("/api/users", {
  method: "GET",
  headers: {
    Authorization: "Bearer token123",
    Accept: "application/json",
  },
});
```

### 2.2 POST 请求

```js
// 发送 JSON 数据
const response = await fetch("/api/users", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    name: "张三",
    email: "zhangsan@example.com",
  }),
});

// 发送 FormData（常用于文件上传）
const formData = new FormData();
formData.append("name", "张三");
formData.append("avatar", fileInput.files[0]);

const response = await fetch("/api/users", {
  method: "POST",
  body: formData, // 不要手动设置 Content-Type，浏览器会自动设置
});
```

### 2.3 PUT 请求（全量更新）

```js
const response = await fetch("/api/users/1", {
  method: "PUT",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    id: 1,
    name: "李四", // 必须提供完整对象
    email: "lisi@example.com",
  }),
});
```

### 2.4 PATCH 请求（部分更新）

```js
const response = await fetch("/api/users/1", {
  method: "PATCH",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    name: "王五", // 只提供需要更新的字段
  }),
});
```

### 2.5 DELETE 请求

```js
const response = await fetch("/api/users/1", {
  method: "DELETE",
  headers: {
    Authorization: "Bearer token123",
  },
});

// 删除后通常没有响应体
if (response.ok) {
  console.log("删除成功");
}
```

### 2.6 请求方法封装

```js
const api = {
  get: (url, params) => {
    const query = params ? "?" + new URLSearchParams(params).toString() : "";
    return fetch(url + query).then(handleResponse);
  },

  post: (url, data) =>
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(handleResponse),

  put: (url, data) =>
    fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(handleResponse),

  patch: (url, data) =>
    fetch(url, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }).then(handleResponse),

  delete: (url) =>
    fetch(url, {
      method: "DELETE",
    }).then(handleResponse),
};

async function handleResponse(response) {
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.message || `HTTP Error: ${response.status}`);
  }
  return response.json();
}
```

---

## 3. 处理 JSON 数据

### 3.1 基本 JSON 解析

```js
const response = await fetch("/api/user/1");
const user = await response.json();
console.log(user.name); // 直接访问解析后的对象
```

### 3.2 类型安全的 JSON 处理

```js
// 使用 TypeScript + Zod 进行运行时类型校验
import { z } from "zod";

const UserSchema = z.object({
  id: z.number(),
  name: z.string(),
  email: z.string().email(),
  age: z.number().optional(),
});

async function fetchUser(id) {
  const response = await fetch(`/api/users/${id}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch user: ${response.status}`);
  }

  const raw = await response.json();

  // 运行时校验
  const user = UserSchema.parse(raw);
  return user; // 类型安全：TypeScript 知道 user 的完整类型
}
```

### 3.3 处理日期等特殊类型

```js
const response = await fetch("/api/events");
const data = await response.json();

// JSON 不支持 Date 类型，需要手动转换
const events = data.map((event) => ({
  ...event,
  createdAt: new Date(event.createdAt),
  updatedAt: new Date(event.updatedAt),
}));
```

### 3.4 处理嵌套数据

```js
// 假设 API 返回：{ data: { users: [...] }, meta: { total: 100 } }
async function fetchUsers() {
  const response = await fetch("/api/users");
  const json = await response.json();

  return {
    users: json.data.users,
    total: json.meta.total,
  };
}
```

---

## 4. 错误处理

### 4.1 区分网络错误和 HTTP 错误

```js
async function safeFetch(url) {
  try {
    const response = await fetch(url);

    // fetch 只有在网络错误时才会 reject
    // HTTP 4xx/5xx 不会抛出异常，需要手动检查
    if (!response.ok) {
      throw new Error(`HTTP Error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      console.log("请求被取消");
      return null;
    }

    if (error instanceof TypeError) {
      // 网络错误（断网、DNS 解析失败等）
      throw new Error("网络连接失败，请检查网络设置");
    }

    throw error;
  }
}
```

### 4.2 try/catch 模式

```jsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadUser() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`/api/users/${userId}`);

        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? "用户不存在"
              : `服务器错误 (${response.status})`
          );
        }

        const data = await response.json();
        setUser(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadUser();
  }, [userId]);

  if (loading) return <div>加载中...</div>;
  if (error) return <div className="error">错误: {error}</div>;

  return <div>{user?.name}</div>;
}
```

### 4.3 response.ok 检查

```js
// 常见的状态码处理
async function handleResponse(response) {
  switch (response.status) {
    case 200:
    case 201:
      return response.json();

    case 204:
      // No Content - 删除成功等场景
      return null;

    case 400:
      const badRequest = await response.json();
      throw new Error(bRequest.message || "请求参数错误");

    case 401:
      // 未授权 - 可能需要跳转登录
      window.location.href = "/login";
      throw new Error("请先登录");

    case 403:
      throw new Error("没有权限执行此操作");

    case 404:
      throw new Error("请求的资源不存在");

    case 429:
      throw new Error("请求过于频繁，请稍后再试");

    case 500:
    case 502:
    case 503:
      throw new Error("服务器内部错误，请稍后再试");

    default:
      throw new Error(`未知错误: ${response.status}`);
  }
}
```

### 4.4 重试机制

```js
async function fetchWithRetry(url, options = {}, retries = 3, delay = 1000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, options);

      // 只对 5xx 错误重试
      if (response.status >= 500 && attempt < retries) {
        await new Promise((r) => setTimeout(r, delay * attempt));
        continue;
      }

      return response;
    } catch (error) {
      // 只对网络错误重试
      if (attempt < retries && error instanceof TypeError) {
        await new Promise((r) => setTimeout(r, delay * attempt));
        continue;
      }
      throw error;
    }
  }
}
```

---

## 5. AbortController 取消请求

### 5.1 基本用法

```js
const controller = new AbortController();

// 发起请求
fetch("/api/data", { signal: controller.signal })
  .then((res) => res.json())
  .then((data) => console.log(data))
  .catch((err) => {
    if (err.name === "AbortError") {
      console.log("请求已取消");
    } else {
      console.error("请求失败:", err);
    }
  });

// 取消请求
controller.abort();
```

### 5.2 在 React 中使用

```jsx
function SearchComponent() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    // 创建 AbortController
    const controller = new AbortController();
    setLoading(true);

    fetch(`/api/search?q=${encodeURIComponent(query)}`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setResults(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error("搜索失败:", err);
          setLoading(false);
        }
        // AbortError 说明是组件卸载或用户输入了新查询，不需要处理
      });

    // 清理：取消上一次的请求
    return () => controller.abort();
  }, [query]);

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="搜索..."
      />
      {loading && <div>搜索中...</div>}
      <ul>
        {results.map((item) => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### 5.3 超时控制

```js
function fetchWithTimeout(url, options = {}, timeout = 5000) {
  const controller = new AbortController();

  const timeoutId = setTimeout(() => controller.abort(), timeout);

  return fetch(url, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(timeoutId));
}

// 使用
try {
  const response = await fetchWithTimeout("/api/data", {}, 3000);
  const data = await response.json();
} catch (err) {
  if (err.name === "AbortError") {
    console.log("请求超时");
  }
}
```

---

## 6. Loading/Error/Data 状态模式

### 6.1 标准三态模式

```jsx
function DataLoader({ url }) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    const controller = new AbortController();

    setState({ data: null, loading: true, error: null });

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setState({ data: null, loading: false, error: err.message });
        }
      });

    return () => controller.abort();
  }, [url]);

  const { data, loading, error } = state;

  if (loading) return <div className="loading">加载中...</div>;
  if (error) return <div className="error">错误: {error}</div>;
  if (!data) return <div className="empty">暂无数据</div>;

  return <DataDisplay data={data} />;
}
```

### 6.2 使用 useReducer 管理复杂状态

```jsx
const initialState = {
  data: null,
  loading: false,
  error: null,
};

function dataReducer(state, action) {
  switch (action.type) {
    case "FETCH_START":
      return { data: null, loading: true, error: null };
    case "FETCH_SUCCESS":
      return { data: action.payload, loading: false, error: null };
    case "FETCH_ERROR":
      return { data: null, loading: false, error: action.payload };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

function UserProfile({ userId }) {
  const [state, dispatch] = useReducer(dataReducer, initialState);

  useEffect(() => {
    const controller = new AbortController();
    dispatch({ type: "FETCH_START" });

    fetch(`/api/users/${userId}`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => dispatch({ type: "FETCH_SUCCESS", payload: data }))
      .catch((err) => {
        if (err.name !== "AbortError") {
          dispatch({ type: "FETCH_ERROR", payload: err.message });
        }
      });

    return () => controller.abort();
  }, [userId]);

  const { data, loading, error } = state;

  if (loading) return <Skeleton />;
  if (error) return <ErrorDisplay message={error} />;
  return <UserCard user={data} />;
}
```

---

## 7. 自定义 useFetch Hook

### 7.1 基础版本

```jsx
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();

    setLoading(true);
    setError(null);

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [url]);

  return { data, loading, error };
}

// 使用
function UserList() {
  const { data: users, loading, error } = useFetch("/api/users");

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error}</div>;

  return (
    <ul>
      {users.map((u) => (
        <li key={u.id}>{u.name}</li>
      ))}
    </ul>
  );
}
```

### 7.2 进阶版本 - 支持 options 和 refetch

```jsx
function useFetch(url, options = {}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 用于触发重新请求
  const [refetchIndex, setRefetchIndex] = useState(0);

  // 稳定的 options 引用
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const refetch = useCallback(() => {
    setRefetchIndex((i) => i + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    setLoading(true);
    setError(null);

    const fetchData = async () => {
      try {
        const response = await fetch(url, {
          ...optionsRef.current,
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP Error: ${response.status}`);
        }

        const result = await response.json();

        if (!cancelled) {
          setData(result);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled && err.name !== "AbortError") {
          setError(err.message);
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [url, refetchIndex]);

  return { data, loading, error, refetch };
}

// 使用
function UserProfile({ userId }) {
  const {
    data: user,
    loading,
    error,
    refetch,
  } = useFetch(`/api/users/${userId}`);

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error}</div>;

  return (
    <div>
      <h1>{user.name}</h1>
      <button onClick={refetch}>刷新</button>
    </div>
  );
}
```

### 7.3 完整版本 - 支持 POST/PUT/DELETE

```jsx
function useApi(baseUrl) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(
    async (endpoint, options = {}) => {
      const controller = new AbortController();
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${baseUrl}${endpoint}`, {
          ...options,
          signal: controller.signal,
          headers: {
            "Content-Type": "application/json",
            ...options.headers,
          },
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.message || `HTTP ${response.status}`);
        }

        const data = await response.json();
        setLoading(false);
        return data;
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(err.message);
          setLoading(false);
        }
        throw err;
      }
    },
    [baseUrl]
  );

  const get = useCallback(
    (endpoint) => request(endpoint),
    [request]
  );

  const post = useCallback(
    (endpoint, body) =>
      request(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    [request]
  );

  const put = useCallback(
    (endpoint, body) =>
      request(endpoint, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    [request]
  );

  const del = useCallback(
    (endpoint) =>
      request(endpoint, { method: "DELETE" }),
    [request]
  );

  return { loading, error, get, post, put, del };
}

// 使用
function UserManager() {
  const { loading, error, get, post, del } = useApi("/api");

  const handleCreate = async () => {
    const newUser = await post("/users", { name: "新用户" });
    console.log("创建成功:", newUser);
  };

  const handleDelete = async (id) => {
    await del(`/users/${id}`);
    console.log("删除成功");
  };

  return (
    <div>
      <button onClick={handleCreate} disabled={loading}>
        创建用户
      </button>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
```

---

## 8. 请求去重 (Request Deduplication)

### 8.1 问题背景

当多个组件同时请求同一个 URL 时，会产生重复的网络请求：

```jsx
// 两个组件同时挂载，发出两次相同的请求
function Page() {
  return (
    <div>
      <UserProfile userId="1" />
      <UserPosts userId="1" /> // 和上面请求同一个用户
    </div>
  );
}
```

### 8.2 手动去重方案

```js
// 请求缓存 Map
const pendingRequests = new Map();

function dedupedFetch(url, options = {}) {
  // 生成缓存 key
  const key = `${options.method || "GET"}:${url}`;

  // 如果有正在进行的请求，直接返回同一个 Promise
  if (pendingRequests.has(key)) {
    return pendingRequests.get(key);
  }

  // 发起请求
  const promise = fetch(url, options)
    .then((res) => res.json())
    .finally(() => {
      // 请求完成后从缓存中移除
      pendingRequests.delete(key);
    });

  // 缓存 Promise
  pendingRequests.set(key, promise);

  return promise;
}
```

### 8.3 带缓存的请求

```js
const cache = new Map();
const pendingRequests = new Map();

async function cachedFetch(url, options = {}, ttl = 60000) {
  const key = `${options.method || "GET"}:${url}`;

  // 检查缓存
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < ttl) {
    return cached.data;
  }

  // 检查是否有正在进行的请求
  if (pendingRequests.has(key)) {
    return pendingRequests.get(key);
  }

  // 发起请求
  const promise = fetch(url, options)
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => {
      cache.set(key, { data, timestamp: Date.now() });
      pendingRequests.delete(key);
      return data;
    })
    .catch((err) => {
      pendingRequests.delete(key);
      throw err;
    });

  pendingRequests.set(key, promise);
  return promise;
}

// 使用
const users = await cachedFetch("/api/users"); // 发起请求
const users2 = await cachedFetch("/api/users"); // 返回缓存（60秒内）
```

### 8.4 React 中的请求去重

```jsx
// 结合 useFetch 使用去重
const globalCache = new Map();

function useFetchDeduped(url) {
  const [data, setData] = useState(() => {
    const cached = globalCache.get(url);
    return cached ?? null;
  });
  const [loading, setLoading] = useState(!globalCache.has(url));
  const [error, setError] = useState(null);

  useEffect(() => {
    // 如果已有缓存，不需要重新请求
    if (globalCache.has(url)) {
      setData(globalCache.get(url));
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((result) => {
        globalCache.set(url, result);
        setData(result);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => controller.abort();
  }, [url]);

  return { data, loading, error };
}
```

---

## 小结

- `fetch()` 是浏览器原生的网络请求 API，返回 Promise
- `fetch()` 只有在网络错误时才会 reject，HTTP 4xx/5xx 需要通过 `response.ok` 检测
- 使用 `AbortController` 在组件卸载时取消请求，防止内存泄漏
- Loading/Error/Data 三态模式是 React 数据请求的标准模式
- 自定义 `useFetch` Hook 可以封装通用的数据请求逻辑
- 请求去重可以避免多个组件同时请求相同数据的浪费
