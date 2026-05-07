# Ch11-03 Axios 与请求封装

## 目录

1. [为什么选择 Axios](#1-为什么选择-axios)
2. [创建 Axios 实例](#2-创建-axios-实例)
3. [请求拦截器](#3-请求拦截器)
4. [响应拦截器](#4-响应拦截器)
5. [错误处理中间件](#5-错误处理中间件)
6. [请求重试](#6-请求重试)
7. [请求取消](#7-请求取消)
8. [文件上传与下载](#8-文件上传与下载)
9. [Base URL 与环境配置](#9-base-url-与环境配置)
10. [完整封装示例](#10-完整封装示例)

---

## 1. 为什么选择 Axios

### 1.1 Axios vs fetch 对比

| 特性 | fetch | Axios |
|------|-------|-------|
| 浏览器支持 | 现代浏览器 | 现代浏览器 + IE11 (polyfill) |
| JSON 自动解析 | 需手动 `res.json()` | 自动解析 |
| 请求/响应拦截器 | 不支持 | 内置支持 |
| 超时控制 | 需配合 AbortController | 内置 `timeout` 配置 |
| 请求取消 | AbortController | CancelToken / AbortController |
| XSRF 防护 | 需手动处理 | 内置支持 |
| 上传进度 | 需手动实现 | `onUploadProgress` 回调 |
| 下载进度 | 需手动实现 | `onDownloadProgress` 回调 |
| 错误处理 | 需检查 `res.ok` | 自动 reject 4xx/5xx |
| 请求去重 | 需手动实现 | 可通过拦截器实现 |

### 1.2 安装

```bash
npm install axios
```

### 1.3 基本使用

```js
import axios from "axios";

// GET 请求
const response = await axios.get("/api/users");
console.log(response.data); // 已经自动解析为 JS 对象

// POST 请求
const response = await axios.post("/api/users", {
  name: "张三",
  email: "zhangsan@example.com",
});

// 带参数的 GET
const response = await axios.get("/api/users", {
  params: { page: 1, limit: 10 },
});
// 实际请求: /api/users?page=1&limit=10
```

---

## 2. 创建 Axios 实例

### 2.1 为什么要创建实例

创建 Axios 实例可以：
- 为不同的 API 服务配置不同的 base URL
- 为不同的认证方式配置不同的拦截器
- 保持配置的隔离性

### 2.2 基本实例创建

```js
import axios from "axios";

// 创建实例
const api = axios.create({
  baseURL: "https://api.example.com",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// 使用
const users = await api.get("/users");
const newUser = await api.post("/users", { name: "张三" });
```

### 2.3 多实例场景

```js
// 用户相关 API
const userApi = axios.create({
  baseURL: "https://api.example.com/v1",
  timeout: 10000,
});

// 文件服务 API（可能有不同的 base URL）
const fileApi = axios.create({
  baseURL: "https://files.example.com",
  timeout: 30000, // 文件上传可能需要更长超时
});

// 第三方 API
const thirdPartyApi = axios.create({
  baseURL: "https://third-party.com/api",
  timeout: 5000,
  headers: {
    "X-API-Key": "third-party-key",
  },
});
```

---

## 3. 请求拦截器

### 3.1 添加认证 Token

```js
api.interceptors.request.use(
  (config) => {
    // 从 localStorage 获取 token
    const token = localStorage.getItem("access_token");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);
```

### 3.2 请求日志

```js
api.interceptors.request.use(
  (config) => {
    // 添加请求 ID 用于追踪
    config.metadata = { startTime: Date.now() };
    config.headers["X-Request-ID"] = crypto.randomUUID();

    console.log(
      `[API Request] ${config.method?.toUpperCase()} ${config.url}`,
      { params: config.params, data: config.data }
    );

    return config;
  },
  (error) => {
    console.error("[API Request Error]", error);
    return Promise.reject(error);
  }
);
```

### 3.3 请求参数转换

```js
api.interceptors.request.use((config) => {
  // 自动将 params 对象中的 undefined/null 值过滤掉
  if (config.params) {
    config.params = Object.fromEntries(
      Object.entries(config.params).filter(
        ([_, value]) => value !== undefined && value !== null
      )
    );
  }

  return config;
});
```

### 3.4 请求签名

```js
import CryptoJS from "crypto-js";

api.interceptors.request.use((config) => {
  const timestamp = Date.now().toString();
  const nonce = Math.random().toString(36).substring(2);

  // 生成签名
  const signStr = `${config.method}${config.url}${timestamp}${nonce}`;
  const signature = CryptoJS.HmacSHA256(signStr, "secret-key").toString();

  config.headers["X-Timestamp"] = timestamp;
  config.headers["X-Nonce"] = nonce;
  config.headers["X-Signature"] = signature;

  return config;
});
```

---

## 4. 响应拦截器

### 4.1 统一处理响应数据

```js
api.interceptors.response.use(
  (response) => {
    // 提取实际数据（假设后端返回 { code: 0, data: ..., message: "ok" }）
    const { code, data, message } = response.data;

    if (code === 0) {
      return data; // 直接返回业务数据
    }

    // 业务错误
    return Promise.reject(new Error(message || "请求失败"));
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 使用时直接拿到 data
const users = await api.get("/users"); // users 就是业务数据
```

### 4.2 响应日志

```js
api.interceptors.response.use(
  (response) => {
    const duration = Date.now() - response.config.metadata.startTime;
    console.log(
      `[API Response] ${response.status} ${response.config.url} (${duration}ms)`
    );
    return response;
  },
  (error) => {
    if (error.config?.metadata) {
      const duration = Date.now() - error.config.metadata.startTime;
      console.error(
        `[API Error] ${error.config.url} (${duration}ms)`,
        error.message
      );
    }
    return Promise.reject(error);
  }
);
```

### 4.3 Token 刷新

```js
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // 如果是 401 且不是刷新 token 的请求
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // 如果正在刷新，将请求加入队列
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return api(originalRequest);
          })
          .catch((err) => Promise.reject(err));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        const { data } = await axios.post("/api/auth/refresh", {
          refreshToken,
        });

        const newToken = data.accessToken;
        localStorage.setItem("access_token", newToken);

        // 更新默认 headers
        api.defaults.headers.common.Authorization = `Bearer ${newToken}`;
        originalRequest.headers.Authorization = `Bearer ${newToken}`;

        // 处理队列中的请求
        processQueue(null, newToken);

        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);

        // 刷新失败，清除 token 并跳转登录
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        window.location.href = "/login";

        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);
```

---

## 5. 错误处理中间件

### 5.1 统一错误处理

```js
// 错误处理函数
function handleError(error) {
  if (error.response) {
    // 服务器返回了错误状态码
    const { status, data } = error.response;

    switch (status) {
      case 400:
        return { type: "VALIDATION_ERROR", message: data.message || "请求参数错误" };
      case 401:
        return { type: "AUTH_ERROR", message: "请先登录" };
      case 403:
        return { type: "FORBIDDEN", message: "没有权限执行此操作" };
      case 404:
        return { type: "NOT_FOUND", message: "请求的资源不存在" };
      case 422:
        return { type: "VALIDATION_ERROR", message: data.message || "数据校验失败" };
      case 429:
        return { type: "RATE_LIMIT", message: "请求过于频繁，请稍后再试" };
      case 500:
        return { type: "SERVER_ERROR", message: "服务器内部错误" };
      default:
        return { type: "UNKNOWN_ERROR", message: `未知错误 (${status})` };
    }
  } else if (error.request) {
    // 请求已发出但没有收到响应
    if (error.code === "ECONNABORTED") {
      return { type: "TIMEOUT", message: "请求超时，请检查网络连接" };
    }
    if (error.code === "ERR_CANCELED") {
      return { type: "CANCELED", message: "请求已取消" };
    }
    return { type: "NETWORK_ERROR", message: "网络连接失败" };
  } else {
    // 请求配置出错
    return { type: "CONFIG_ERROR", message: error.message };
  }
}

// 使用拦截器统一处理
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const errInfo = handleError(error);

    // 统一通知
    notification.error({
      message: errInfo.message,
      description: error.response?.data?.detail,
    });

    // 返回一个包含错误信息的 rejected Promise
    return Promise.reject(errInfo);
  }
);
```

### 5.2 自定义错误类

```js
class ApiError extends Error {
  constructor(type, message, status, data) {
    super(message);
    this.name = "ApiError";
    this.type = type;
    this.status = status;
    this.data = data;
  }

  static fromAxiosError(error) {
    if (error.response) {
      const { status, data } = error.response;
      return new ApiError(
        "HTTP_ERROR",
        data.message || `HTTP ${status}`,
        status,
        data
      );
    }
    if (error.code === "ECONNABORTED") {
      return new ApiError("TIMEOUT", "请求超时");
    }
    return new ApiError("NETWORK_ERROR", "网络错误");
  }
}

// 在拦截器中使用
api.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(ApiError.fromAxiosError(error))
);
```

---

## 6. 请求重试

### 6.1 手动实现重试

```js
async function requestWithRetry(config, retries = 3, delay = 1000) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await api(config);
    } catch (error) {
      const status = error.response?.status;

      // 只重试 5xx 和网络错误
      const shouldRetry =
        attempt < retries &&
        (status >= 500 || error.type === "NETWORK_ERROR" || error.type === "TIMEOUT");

      if (!shouldRetry) {
        throw error;
      }

      // 指数退避
      const waitTime = delay * Math.pow(2, attempt - 1);
      await new Promise((r) => setTimeout(r, waitTime));

      console.log(`请求重试 (${attempt}/${retries}): ${config.url}`);
    }
  }
}

// 使用
const data = await requestWithRetry(
  { method: "GET", url: "/api/important-data" },
  3,
  1000
);
```

### 6.2 使用 axios-retry 插件

```bash
npm install axios-retry
```

```js
import axiosRetry from "axios-retry";

axiosRetry(api, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay, // 指数退避
  retryCondition: (error) => {
    // 只对网络错误和 5xx 重试
    return (
      axiosRetry.isNetworkOrIdempotentRequestError(error) ||
      error.response?.status >= 500
    );
  },
  onRetry: (retryCount, error, requestConfig) => {
    console.log(`重试请求 (${retryCount}): ${requestConfig.url}`);
  },
});
```

---

## 7. 请求取消

### 7.1 AbortController 方式（推荐）

```js
// 在 React 中使用
function SearchResults({ query }) {
  const [results, setResults] = useState([]);
  const controllerRef = useRef(null);

  useEffect(() => {
    // 取消上一次请求
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    const controller = new AbortController();
    controllerRef.current = controller;

    api
      .get("/api/search", {
        params: { q: query },
        signal: controller.signal,
      })
      .then((res) => setResults(res.data))
      .catch((err) => {
        if (err.code !== "ERR_CANCELED") {
          console.error(err);
        }
      });

    return () => controller.abort();
  }, [query]);

  return <ResultList results={results} />;
}
```

### 7.2 CancelToken 方式（旧版，不推荐）

```js
// CancelToken 已被废弃，但旧项目中可能还在使用
const source = axios.CancelToken.source();

api.get("/api/data", {
  cancelToken: source.token,
}).catch((err) => {
  if (axios.isCancel(err)) {
    console.log("请求已取消:", err.message);
  }
});

// 取消请求
source.cancel("用户取消了请求");
```

### 7.3 批量取消

```js
class RequestManager {
  constructor() {
    this.controllers = new Map();
  }

  create(key) {
    // 取消同 key 的已有请求
    this.cancel(key);

    const controller = new AbortController();
    this.controllers.set(key, controller);
    return controller.signal;
  }

  cancel(key) {
    const controller = this.controllers.get(key);
    if (controller) {
      controller.abort();
      this.controllers.delete(key);
    }
  }

  cancelAll() {
    this.controllers.forEach((controller) => controller.abort());
    this.controllers.clear();
  }
}

const requestManager = new RequestManager();

// 使用
api.get("/api/search", {
  signal: requestManager.create("search"),
});

// 组件卸载时取消所有请求
useEffect(() => {
  return () => requestManager.cancelAll();
}, []);
```

---

## 8. 文件上传与下载

### 8.1 文件上传

```js
// 单文件上传
async function uploadFile(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/api/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (progressEvent) => {
      const percent = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress?.(percent);
    },
  });

  return response.data;
}

// 多文件上传
async function uploadFiles(files, onProgress) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await api.post("/api/upload/multiple", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (progressEvent) => {
      const percent = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress?.(percent);
    },
  });

  return response.data;
}
```

### 8.2 React 文件上传组件

```jsx
function FileUploader() {
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setProgress(0);

    try {
      const data = await uploadFile(file, setProgress);
      setResult(data);
    } catch (err) {
      console.error("上传失败:", err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleUpload} disabled={uploading} />
      {uploading && (
        <div>
          <progress value={progress} max="100" />
          <span>{progress}%</span>
        </div>
      )}
      {result && <div>上传成功: {result.url}</div>}
    </div>
  );
}
```

### 8.3 文件下载

```js
async function downloadFile(url, filename) {
  const response = await api.get(url, {
    responseType: "blob", // 关键：指定响应类型为 blob
  });

  // 创建下载链接
  const blob = new Blob([response.data]);
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();

  // 清理
  document.body.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
}

// 带进度的下载
async function downloadWithProgress(url, filename, onProgress) {
  const response = await api.get(url, {
    responseType: "blob",
    onDownloadProgress: (progressEvent) => {
      if (progressEvent.total) {
        const percent = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress?.(percent);
      }
    },
  });

  const blob = new Blob([response.data]);
  const link = document.createElement("a");
  link.href = window.URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(link.href);
}
```

---

## 9. Base URL 与环境配置

### 9.1 环境变量配置

```js
// .env.development
VITE_API_BASE_URL=http://localhost:3001/api

// .env.production
VITE_API_BASE_URL=https://api.example.com/api
```

```js
// src/api/client.js
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 10000,
});
```

### 9.2 多环境配置

```js
const envConfig = {
  development: {
    baseURL: "http://localhost:3001/api",
    timeout: 30000,
  },
  staging: {
    baseURL: "https://staging-api.example.com/api",
    timeout: 15000,
  },
  production: {
    baseURL: "https://api.example.com/api",
    timeout: 10000,
  },
};

const currentEnv = import.meta.env.MODE || "development";
const config = envConfig[currentEnv];

const api = axios.create(config);
```

---

## 10. 完整封装示例

### 10.1 项目结构

```
src/api/
  ├── client.js          # Axios 实例和拦截器
  ├── auth.js            # 认证相关 API
  ├── user.js            # 用户相关 API
  ├── error.js           # 错误处理
  └── index.js           # 统一导出
```

### 10.2 client.js

```js
import axios from "axios";

// 创建 Axios 实例
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// ========== 请求拦截器 ==========

client.interceptors.request.use(
  (config) => {
    // 1. 添加 Token
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // 2. 记录请求时间（用于日志）
    config.metadata = { startTime: Date.now() };

    // 3. 过滤空参数
    if (config.params) {
      config.params = Object.fromEntries(
        Object.entries(config.params).filter(
          ([, v]) => v !== undefined && v !== null && v !== ""
        )
      );
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// ========== 响应拦截器 ==========

client.interceptors.response.use(
  (response) => {
    // 日志
    const duration = Date.now() - response.config.metadata.startTime;
    if (import.meta.env.DEV) {
      console.log(
        `[API] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status} (${duration}ms)`
      );
    }

    // 统一提取业务数据
    const { code, data, message } = response.data;
    if (code === 0 || code === undefined) {
      return data ?? response.data;
    }

    // 业务错误
    const error = new Error(message || "请求失败");
    error.code = code;
    return Promise.reject(error);
  },
  (error) => {
    // 统一错误处理
    if (error.response) {
      const { status } = error.response;

      if (status === 401) {
        localStorage.removeItem("access_token");
        window.location.href = "/login";
      }

      if (status === 403) {
        console.warn("没有权限");
      }

      if (status === 500) {
        console.error("服务器错误");
      }
    }

    return Promise.reject(error);
  }
);

export default client;
```

### 10.3 user.js

```js
import client from "./client";

export const userApi = {
  // 获取用户列表
  getList: (params) => client.get("/users", { params }),

  // 获取单个用户
  getById: (id) => client.get(`/users/${id}`),

  // 创建用户
  create: (data) => client.post("/users", data),

  // 更新用户
  update: (id, data) => client.put(`/users/${id}`, data),

  // 部分更新
  patch: (id, data) => client.patch(`/users/${id}`, data),

  // 删除用户
  delete: (id) => client.delete(`/users/${id}`),

  // 上传头像
  uploadAvatar: (id, file, onProgress) => {
    const formData = new FormData();
    formData.append("avatar", file);

    return client.post(`/users/${id}/avatar`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        const percent = Math.round((e.loaded * 100) / e.total);
        onProgress?.(percent);
      },
    });
  },
};
```

### 10.4 error.js

```js
export class ApiError extends Error {
  constructor({ type, message, status, data }) {
    super(message);
    this.type = type;
    this.status = status;
    this.data = data;
  }

  static fromAxiosError(error) {
    if (error.response) {
      return new ApiError({
        type: "HTTP_ERROR",
        message: error.response.data?.message || `HTTP ${error.response.status}`,
        status: error.response.status,
        data: error.response.data,
      });
    }

    if (error.code === "ECONNABORTED") {
      return new ApiError({ type: "TIMEOUT", message: "请求超时" });
    }

    if (error.code === "ERR_CANCELED") {
      return new ApiError({ type: "CANCELED", message: "请求已取消" });
    }

    return new ApiError({ type: "NETWORK_ERROR", message: "网络连接失败" });
  }
}
```

### 10.5 在 React 中使用

```jsx
import { useState, useEffect } from "react";
import { userApi } from "../api";

function UserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 加载用户列表
  const loadUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await userApi.getList({ page: 1, limit: 20 });
      setUsers(data.list);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  // 删除用户
  const handleDelete = async (id) => {
    try {
      await userApi.delete(id);
      setUsers((prev) => prev.filter((u) => u.id !== id));
    } catch (err) {
      alert(err.message);
    }
  };

  // 创建用户
  const handleCreate = async (userData) => {
    try {
      const newUser = await userApi.create(userData);
      setUsers((prev) => [...prev, newUser]);
    } catch (err) {
      alert(err.message);
    }
  };

  if (loading) return <div>加载中...</div>;
  if (error) return <div>错误: {error}</div>;

  return (
    <div>
      <button onClick={() => handleCreate({ name: "新用户" })}>添加用户</button>
      <ul>
        {users.map((user) => (
          <li key={user.id}>
            {user.name}
            <button onClick={() => handleDelete(user.id)}>删除</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## 小结

- Axios 提供了拦截器、超时控制、请求取消等内置功能，比 fetch 更适合企业级项目
- 创建 Axios 实例可以为不同服务配置不同的 base URL 和拦截器
- 请求拦截器常用于添加 Token、日志、参数转换
- 响应拦截器常用于统一数据提取、错误处理、Token 刷新
- 使用 AbortController 取消请求，避免组件卸载后的状态更新
- 文件上传/下载使用 `multipart/form-data` 和 `responseType: "blob"`
- 按模块组织 API 代码，保持清晰的项目结构
