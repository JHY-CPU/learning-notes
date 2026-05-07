# Ch11-04 SWR 与 React Query

## 目录

1. [SWR 基础](#1-swr-基础)
2. [SWR 缓存与重新验证](#2-swr-缓存与重新验证)
3. [SWR Mutate](#3-swr-mutate)
4. [SWR 错误重试与乐观更新](#4-swr-错误重试与乐观更新)
5. [TanStack Query (React Query) 基础](#5-tanstack-query-react-query-基础)
6. [useQuery 详解](#6-usequery-详解)
7. [useMutation 详解](#7-usemutation-详解)
8. [查询键 (Query Keys)](#8-查询键-query-keys)
9. [缓存时间与陈旧时间](#9-缓存时间与陈旧时间)
10. [预取与无限查询](#10-预取与无限查询)
11. [DevTools](#11-devtools)
12. [SWR vs TanStack Query 选型对比](#12-swr-vs-tanstack-query-选型对比)

---

## 1. SWR 基础

### 1.1 什么是 SWR

SWR（stale-while-revalidate）是 Vercel 开源的数据请求库，名称来自 HTTP 缓存策略：先返回缓存数据（stale），同时在后台发送请求获取最新数据（revalidate）。

```bash
npm install swr
```

### 1.2 基本使用

```jsx
import useSWR from "swr";

// 定义 fetcher（可以是任何返回 Promise 的函数）
const fetcher = (url) => fetch(url).then((res) => res.json());

function UserProfile({ userId }) {
  const { data, error, isLoading } = useSWR(`/api/users/${userId}`, fetcher);

  if (isLoading) return <div>加载中...</div>;
  if (error) return <div>加载失败: {error.message}</div>;

  return (
    <div>
      <h1>{data.name}</h1>
      <p>{data.email}</p>
    </div>
  );
}
```

### 1.3 不同的 Fetcher

```js
// 使用 fetch
const fetcher = (url) => fetch(url).then((res) => res.json());

// 使用 axios
import axios from "axios";
const fetcher = (url) => axios.get(url).then((res) => res.data);

// 带认证的 fetcher
const fetcher = (url) =>
  fetch(url, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`,
    },
  }).then((res) => res.json());

// 全局配置 fetcher
import { SWRConfig } from "swr";

function App() {
  return (
    <SWRConfig value={{ fetcher: (url) => fetch(url).then((r) => r.json()) }}>
      <UserProfile userId="1" />
    </SWRConfig>
  );
}
```

### 1.4 条件请求

```jsx
function UserProfile({ userId }) {
  // key 为 null 时不会发起请求
  const { data } = useSWR(userId ? `/api/users/${userId}` : null, fetcher);

  // 依赖其他请求的结果
  const { data: user } = useSWR(`/api/user`, fetcher);
  const { data: posts } = useSWR(
    user ? `/api/users/${user.id}/posts` : null,
    fetcher
  );

  return <div>{posts?.map((p) => <Post key={p.id} post={p} />)}</div>;
}
```

---

## 2. SWR 缓存与重新验证

### 2.1 缓存机制

```jsx
// 同一个 key 的多个组件共享缓存
function ComponentA() {
  const { data } = useSWR("/api/user", fetcher);
  return <div>Component A: {data?.name}</div>;
}

function ComponentB() {
  const { data } = useSWR("/api/user", fetcher); // 复用缓存，不会重复请求
  return <div>Component B: {data?.name}</div>;
}
```

### 2.2 重新验证配置

```jsx
const { data } = useSWR("/api/user", fetcher, {
  // 聚焦时重新验证
  revalidateOnFocus: true,

  // 重新连接时重新验证
  revalidateOnReconnect: true,

  // 定期重新验证（毫秒）
  refreshInterval: 3000,

  // 页面不可见时也继续刷新
  refreshWhenHidden: false,

  // 离线时停止刷新
  refreshWhenOffline: false,

  // 去抖时间（多次 key 变化只触发一次请求）
  dedupingInterval: 2000,

  // 错误重试间隔
  errorRetryInterval: 5000,

  // 最大重试次数
  errorRetryCount: 3,
});
```

### 2.3 全局配置

```jsx
function App() {
  return (
    <SWRConfig
      value={{
        refreshInterval: 3000,
        revalidateOnFocus: true,
        fetcher: (url) => fetch(url).then((r) => r.json()),
        onError: (error) => {
          if (error.status === 403 || error.status === 404) {
            // 可以在这里处理全局错误
          }
        },
      }}
    >
      <MyApp />
    </SWRConfig>
  );
}
```

---

## 3. SWR Mutate

### 3.1 什么是 Mutate

Mutate 允许你手动更新缓存数据，常用于乐观更新（Optimistic UI）。

```jsx
import useSWR, { mutate } from "swr";

// 方式一：使用 hook 返回的 mutate
function TodoList() {
  const { data: todos, mutate: todosMutate } = useSWR("/api/todos", fetcher);

  const addTodo = async (text) => {
    // 乐观更新
    const newTodo = { id: Date.now(), text, done: false };
    todosMutate([...todos, newTodo], false); // false = 不立即重新验证

    try {
      await fetch("/api/todos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      todosMutate(); // 成功后重新验证
    } catch (error) {
      todosMutate(); // 失败时恢复
    }
  };

  return (
    <div>
      <button onClick={() => addTodo("新任务")}>添加</button>
      {todos?.map((t) => (
        <div key={t.id}>{t.text}</div>
      ))}
    </div>
  );
}

// 方式二：全局 mutate（可以在任何地方调用）
import { mutate } from "swr";

function updateUser(userId, newData) {
  // 乐观更新
  mutate(`/api/users/${userId}`, newData, false);

  // 发送请求
  return fetch(`/api/users/${userId}`, {
    method: "PUT",
    body: JSON.stringify(newData),
  }).then(() => mutate(`/api/users/${userId}`));
}
```

### 3.2 批量 Mutate

```jsx
import { mutate } from "swr";

async function deleteAllTodos() {
  // 乐观更新
  mutate("/api/todos", [], false);

  try {
    await fetch("/api/todos", { method: "DELETE" });
  } catch {
    mutate("/api/todos"); // 恢复
  }
}
```

---

## 4. SWR 错误重试与乐观更新

### 4.1 错误处理

```jsx
const { data, error } = useSWR("/api/user", fetcher, {
  onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
    // 404 不重试
    if (error.status === 404) return;
    // 最多重试 5 次
    if (retryCount >= 5) return;
    // 3 秒后重试
    setTimeout(() => revalidate({ retryCount }), 3000);
  },
});

// 全局错误处理
<SWRConfig
  value={{
    onError: (error, key) => {
      if (error.status === 401) {
        // 跳转到登录页
        router.push("/login");
      }
      // 上报错误到监控服务
      Sentry.captureException(error);
    },
  }}
>
```

### 4.2 完整乐观更新示例

```jsx
function TodoApp() {
  const { data: todos, mutate } = useSWR("/api/todos", fetcher);

  const toggleTodo = async (id) => {
    const todo = todos.find((t) => t.id === id);

    // 1. 乐观更新 UI
    mutate(
      todos.map((t) => (t.id === id ? { ...t, done: !t.done } : t)),
      false
    );

    // 2. 发送请求
    try {
      await fetch(`/api/todos/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ done: !todo.done }),
      });
      // 3. 请求成功，重新验证确保数据一致
      mutate();
    } catch (error) {
      // 4. 请求失败，回滚到原始数据
      mutate(todos);
    }
  };

  return (
    <ul>
      {todos?.map((todo) => (
        <li key={todo.id}>
          <input
            type="checkbox"
            checked={todo.done}
            onChange={() => toggleTodo(todo.id)}
          />
          <span style={{ textDecoration: todo.done ? "line-through" : "none" }}>
            {todo.text}
          </span>
        </li>
      ))}
    </ul>
  );
}
```

---

## 5. TanStack Query (React Query) 基础

### 5.1 安装与配置

```bash
npm install @tanstack/react-query
```

```jsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 分钟
      gcTime: 10 * 60 * 1000, // 10 分钟 (旧版本叫 cacheTime)
      retry: 3,
      refetchOnWindowFocus: true,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MyApp />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}
```

---

## 6. useQuery 详解

### 6.1 基本使用

```jsx
import { useQuery } from "@tanstack/react-query";

function UserProfile({ userId }) {
  const { data, isPending, isError, error, isFetching } = useQuery({
    queryKey: ["users", userId],
    queryFn: () =>
      fetch(`/api/users/${userId}`).then((res) => res.json()),
  });

  if (isPending) return <div>加载中...</div>;
  if (isError) return <div>错误: {error.message}</div>;

  return (
    <div>
      <h1>{data.name}</h1>
      {isFetching && <div>更新中...</div>}
    </div>
  );
}
```

### 6.2 查询状态

```jsx
const {
  data, // 查询结果
  isPending, // 首次加载中（无缓存数据）
  isError, // 查询出错
  error, // 错误对象
  isSuccess, // 查询成功
  isFetching, // 任何后台获取中
  isLoading, // 等同于 isPending（兼容旧版）
  isRefetching, // 后台重新获取中
  isStale, // 数据是否已过期
  isPlaceholderData, // 是否为占位数据
  dataUpdatedAt, // 数据最后更新时间
  errorUpdatedAt, // 错误最后更新时间
  failureCount, // 连续失败次数
} = useQuery({
  queryKey: ["todos"],
  queryFn: fetchTodos,
});
```

### 6.3 高级配置

```jsx
const { data } = useQuery({
  queryKey: ["users", userId],
  queryFn: () => fetchUser(userId),

  // 缓存配置
  staleTime: 5 * 60 * 1000, // 5 分钟内不重新请求
  gcTime: 10 * 60 * 1000, // 10 分钟后清理缓存

  // 重试配置
  retry: 3,
  retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30000),

  // 刷新配置
  refetchOnWindowFocus: true,
  refetchOnReconnect: true,
  refetchInterval: 30000, // 每 30 秒轮询

  // 条件查询
  enabled: !!userId,

  // 选择部分数据
  select: (data) => ({
    ...data,
    fullName: `${data.firstName} ${data.lastName}`,
  }),

  // 占位数据（保持旧数据作为占位）
  placeholderData: keepPreviousData,

  // 初始数据
  initialData: () => getInitialUser(userId),
});
```

---

## 7. useMutation 详解

### 7.1 基本使用

```jsx
import { useMutation, useQueryClient } from "@tanstack/react-query";

function AddTodo() {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (newTodo) =>
      fetch("/api/todos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newTodo),
      }).then((res) => res.json()),

    onSuccess: (data) => {
      // 使相关查询失效，触发重新请求
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },

    onError: (error) => {
      console.error("创建失败:", error.message);
    },

    onSettled: () => {
      // 无论成功失败都执行
      console.log("请求完成");
    },
  });

  const handleSubmit = () => {
    mutation.mutate({ text: "新任务", done: false });
  };

  return (
    <div>
      <button onClick={handleSubmit} disabled={mutation.isPending}>
        {mutation.isPending ? "添加中..." : "添加任务"}
      </button>
      {mutation.isError && <div>错误: {mutation.error.message}</div>}
      {mutation.isSuccess && <div>添加成功!</div>}
    </div>
  );
}
```

### 7.2 乐观更新

```jsx
function TodoList() {
  const queryClient = useQueryClient();

  const { data: todos } = useQuery({
    queryKey: ["todos"],
    queryFn: fetchTodos,
  });

  const toggleMutation = useMutation({
    mutationFn: (todo) =>
      fetch(`/api/todos/${todo.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ done: !todo.done }),
      }),

    // 乐观更新：在请求发出前更新 UI
    onMutate: async (todo) => {
      // 取消正在进行的查询，防止覆盖乐观更新
      await queryClient.cancelQueries({ queryKey: ["todos"] });

      // 保存之前的数据用于回滚
      const previousTodos = queryClient.getQueryData(["todos"]);

      // 乐观更新缓存
      queryClient.setQueryData(["todos"], (old) =>
        old.map((t) => (t.id === todo.id ? { ...t, done: !t.done } : t))
      );

      // 返回 context 用于 onError 回滚
      return { previousTodos };
    },

    // 失败时回滚
    onError: (err, todo, context) => {
      queryClient.setQueryData(["todos"], context.previousTodos);
    },

    // 成功后重新验证
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["todos"] });
    },
  });

  return (
    <ul>
      {todos?.map((todo) => (
        <li key={todo.id}>
          <input
            type="checkbox"
            checked={todo.done}
            onChange={() => toggleMutation.mutate(todo)}
          />
          {todo.text}
        </li>
      ))}
    </ul>
  );
}
```

### 7.3 useMutation 返回值

```jsx
const mutation = useMutation({
  mutationFn: createTodo,
});

// 状态
mutation.isPending; // 请求进行中
mutation.isError; // 请求失败
mutation.isSuccess; // 请求成功
mutation.isIdle; // 未开始
mutation.data; // 成功结果
mutation.error; // 错误对象
mutation.variables; // 传入 mutate 的参数
mutation.reset(); // 重置状态

// 方法
mutation.mutate({ text: "新任务" }); // 异步触发
mutation.mutateAsync({ text: "新任务" }); // 返回 Promise
```

---

## 8. 查询键 (Query Keys)

### 8.1 查询键的作用

查询键用于标识和缓存查询结果，结构为数组：

```js
// 简单键
useQuery({ queryKey: ["todos"], queryFn: fetchTodos });

// 带参数的键
useQuery({ queryKey: ["todos", todoId], queryFn: () => fetchTodo(todoId) });

// 带过滤器的键
useQuery({
  queryKey: ["todos", { status: "done", page: 1 }],
  queryFn: () => fetchTodos({ status: "done", page: 1 }),
});
```

### 8.2 查询键的匹配规则

```js
// 精确匹配
queryClient.invalidateQueries({ queryKey: ["todos", 1] });
// 匹配: ["todos", 1]
// 不匹配: ["todos", 2]、["todos", 1, "detail"]

// 部分匹配（前缀匹配）
queryClient.invalidateQueries({ queryKey: ["todos"] });
// 匹配: ["todos"]、["todos", 1]、["todos", { status: "done" }]
// 不匹配: ["todo"]、["users"]
```

### 8.3 常用的查询键模式

```js
// 按资源分组
const todoKeys = {
  all: ["todos"],
  lists: () => [...todoKeys.all, "list"],
  list: (filters) => [...todoKeys.lists(), filters],
  details: () => [...todoKeys.all, "detail"],
  detail: (id) => [...todoKeys.details(), id],
};

// 使用
useQuery({ queryKey: todoKeys.lists(), queryFn: fetchAllTodos });
useQuery({ queryKey: todoKeys.list({ status: "done" }), queryFn: fetchFilteredTodos });
useQuery({ queryKey: todoKeys.detail(1), queryFn: () => fetchTodo(1) });

// 使所有 todo 相关查询失效
queryClient.invalidateQueries({ queryKey: todoKeys.all });

// 只使列表查询失效
queryClient.invalidateQueries({ queryKey: todoKeys.lists() });
```

---

## 9. 缓存时间与陈旧时间

### 9.1 核心概念

```
|--- staleTime ---|--- "stale" 期间 ---|--- gcTime ---|
|   新鲜数据       |   数据可用但标记过期  |  缓存被清理    |
|   不重新请求     |   触发重新请求       |               |
```

- **staleTime**：数据保持"新鲜"的时间。在此期间不会发起新的请求。
- **gcTime**（原 cacheTime）：不活跃的缓存数据保留的时间。超过此时间后从内存中清除。

### 9.2 配置示例

```jsx
// 场景一：频繁变化的数据 - 短 staleTime
useQuery({
  queryKey: ["stock-price"],
  queryFn: fetchStockPrice,
  staleTime: 1000, // 1 秒后就认为过期
  refetchInterval: 5000, // 每 5 秒轮询
});

// 场景二：几乎不变的数据 - 长 staleTime
useQuery({
  queryKey: ["countries"],
  queryFn: fetchCountries,
  staleTime: Infinity, // 永不认为过期（只请求一次）
  gcTime: 24 * 60 * 60 * 1000, // 缓存保留 24 小时
});

// 场景三：中等变化频率
useQuery({
  queryKey: ["user-profile"],
  queryFn: fetchUserProfile,
  staleTime: 5 * 60 * 1000, // 5 分钟
});
```

### 9.3 全局默认配置

```js
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000, // 默认 1 分钟
      gcTime: 5 * 60 * 1000, // 默认 5 分钟
    },
  },
});
```

---

## 10. 预取与无限查询

### 10.1 预取 (Prefetching)

```jsx
const queryClient = useQueryClient();

// 手动预取
const prefetchUser = async (userId) => {
  await queryClient.prefetchQuery({
    queryKey: ["users", userId],
    queryFn: () => fetchUser(userId),
    staleTime: 60000,
  });
};

// 在 hover 时预取
function UserLink({ userId }) {
  return (
    <Link
      to={`/users/${userId}`}
      onMouseEnter={() => prefetchUser(userId)}
    >
      查看用户
    </Link>
  );
}
```

### 10.2 无限查询 (Infinite Queries)

```jsx
import { useInfiniteQuery } from "@tanstack/react-query";

function InfiniteTodoList() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
  } = useInfiniteQuery({
    queryKey: ["todos"],
    queryFn: ({ pageParam = 1 }) =>
      fetch(`/api/todos?page=${pageParam}&limit=10`).then((res) => res.json()),
    getNextPageParam: (lastPage) => {
      // lastPage 是 queryFn 的返回值
      // 返回下一页的页码，如果没有更多数据则返回 undefined
      return lastPage.hasNextPage ? lastPage.page + 1 : undefined;
    },
    initialPageParam: 1,
  });

  if (isLoading) return <div>加载中...</div>;

  // data.pages 是一个数组，包含所有已加载的页面数据
  const allTodos = data.pages.flatMap((page) => page.todos);

  return (
    <div>
      {allTodos.map((todo) => (
        <div key={todo.id}>{todo.text}</div>
      ))}

      <button
        onClick={() => fetchNextPage()}
        disabled={!hasNextPage || isFetchingNextPage}
      >
        {isFetchingNextPage
          ? "加载中..."
          : hasNextPage
          ? "加载更多"
          : "没有更多了"}
      </button>
    </div>
  );
}
```

### 10.3 无限滚动加载

```jsx
function InfiniteScrollList() {
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage } =
    useInfiniteQuery({
      queryKey: ["items"],
      queryFn: ({ pageParam }) => fetchItems(pageParam),
      getNextPageParam: (lastPage) => lastPage.nextCursor,
      initialPageParam: null,
    });

  const observerRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { threshold: 0.1 }
    );

    if (observerRef.current) {
      observer.observe(observerRef.current);
    }

    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const items = data?.pages.flatMap((page) => page.items) ?? [];

  return (
    <div>
      {items.map((item) => (
        <div key={item.id}>{item.name}</div>
      ))}
      <div ref={observerRef} style={{ height: 20 }}>
        {isFetchingNextPage && "加载中..."}
      </div>
    </div>
  );
}
```

---

## 11. DevTools

### 11.1 TanStack Query DevTools

```bash
npm install @tanstack/react-query-devtools
```

```jsx
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MyApp />
      {/* 开发环境下显示 DevTools */}
      {import.meta.env.DEV && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  );
}
```

DevTools 功能：
- 查看所有活跃查询及其状态
- 查看缓存数据内容
- 手动触发重新请求
- 查看查询的时间线
- 标记过期的查询

---

## 12. SWR vs TanStack Query 选型对比

### 12.1 功能对比

| 特性 | SWR | TanStack Query |
|------|-----|---------------|
| 包大小 | ~4.2kB | ~13kB |
| 缓存策略 | stale-while-revalidate | stale-time + gc-time |
| 查询键 | 字符串/数组 | 数组 |
| Mutations | 手动 mutate | 内置 useMutation |
| 乐观更新 | 手动实现 | 内置 onMutate/onError 回调 |
| 无限查询 | useSWRInfinite | useInfiniteQuery |
| 预取 | mutate 预取 | prefetchQuery |
| DevTools | 社区版 | 官方 DevTools |
| 框架支持 | React | React/Vue/Solid/Angular/Svelte |
| SSR 支持 | 基础 | 更完善 |
| Mutation 状态 | 需手动管理 | 内置 isPending/isError/isSuccess |
| 查询取消 | 不支持 | 内置 signal 支持 |

### 12.2 适用场景

**选择 SWR 的场景：**
- 项目较小，主要需要 GET 请求的缓存
- 追求极小的包体积
- 使用 Vercel / Next.js 技术栈
- 数据获取逻辑简单

**选择 TanStack Query 的场景：**
- 项目较大，需要完善的 mutation 管理
- 需要乐观更新、查询失效等高级功能
- 需要更好的 DevTools 支持
- 需要跨框架使用
- 需要完善的 TypeScript 支持

### 12.3 代码对比

```jsx
// ========== SWR 方案 ==========
import useSWR from "swr";

function Todos() {
  const { data: todos, mutate } = useSWR("/api/todos", fetcher);

  const addTodo = async (text) => {
    // 手动乐观更新
    mutate([...(todos || []), { id: Date.now(), text }], false);
    await fetch("/api/todos", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    mutate(); // 重新验证
  };

  return <button onClick={() => addTodo("新任务")}>添加</button>;
}

// ========== TanStack Query 方案 ==========
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

function Todos() {
  const queryClient = useQueryClient();
  const { data: todos } = useQuery({
    queryKey: ["todos"],
    queryFn: fetchTodos,
  });

  const addMutation = useMutation({
    mutationFn: (text) =>
      fetch("/api/todos", {
        method: "POST",
        body: JSON.stringify({ text }),
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["todos"] }),
  });

  return (
    <button onClick={() => addMutation.mutate("新任务")} disabled={addMutation.isPending}>
      添加
    </button>
  );
}
```

---

## 小结

- SWR 和 TanStack Query 都是优秀的数据请求库，核心思想是缓存 + 后台重新验证
- SWR 更轻量简洁，TanStack Query 功能更丰富
- 两者都支持条件请求、缓存共享、乐观更新
- TanStack Query 提供更完善的 mutation 管理和 DevTools
- 选择取决于项目复杂度、团队偏好和技术栈
- 对于复杂的企业级应用，推荐 TanStack Query
