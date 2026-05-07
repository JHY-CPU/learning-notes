# React 项目实战架构

## 一、项目目录结构

### 1.1 Feature-based 结构（推荐）

```
src/
├── app/                          # 应用入口与配置
│   ├── providers.tsx             # 全局 Provider
│   ├── router.tsx                # 路由配置
│   └── App.tsx                   # 根组件
│
├── features/                     # 按功能模块组织
│   ├── auth/
│   │   ├── api/                  # 该功能的 API 请求
│   │   │   ├── auth.api.ts
│   │   │   └── auth.types.ts
│   │   ├── components/           # 功能专属组件
│   │   │   ├── LoginForm.tsx
│   │   │   ├── RegisterForm.tsx
│   │   │   └── AuthGuard.tsx
│   │   ├── hooks/                # 功能专属 hooks
│   │   │   ├── useAuth.ts
│   │   │   └── useLogin.ts
│   │   ├── stores/               # 功能专属状态
│   │   │   └── authStore.ts
│   │   ├── utils/                # 功能专属工具
│   │   │   └── token.ts
│   │   └── index.ts              # Barrel export
│   │
│   ├── products/
│   │   ├── api/
│   │   │   ├── products.api.ts
│   │   │   └── products.types.ts
│   │   ├── components/
│   │   │   ├── ProductCard.tsx
│   │   │   ├── ProductList.tsx
│   │   │   ├── ProductDetail.tsx
│   │   │   ├── ProductFilters.tsx
│   │   │   └── ProductForm.tsx
│   │   ├── hooks/
│   │   │   ├── useProducts.ts
│   │   │   └── useProductFilter.ts
│   │   └── index.ts
│   │
│   ├── cart/
│   └── orders/
│
├── components/                   # 共享通用组件
│   ├── ui/                       # UI 基础组件（Design System）
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   ├── Button.stories.tsx
│   │   │   ├── Button.module.css
│   │   │   └── index.ts
│   │   ├── Input/
│   │   ├── Modal/
│   │   ├── Table/
│   │   ├── Toast/
│   │   └── index.ts              # 统一导出
│   ├── layout/                   # 布局组件
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Footer.tsx
│   │   └── MainLayout.tsx
│   └── common/                   # 业务通用组件
│       ├── ErrorBoundary.tsx
│       ├── LoadingSpinner.tsx
│       └── PageTitle.tsx
│
├── hooks/                        # 全局通用 hooks
│   ├── useDebounce.ts
│   ├── useLocalStorage.ts
│   ├── useMediaQuery.ts
│   ├── useClickOutside.ts
│   └── index.ts
│
├── lib/                          # 第三方库配置
│   ├── axios.ts                  # Axios 实例
│   ├── queryClient.ts            # React Query 客户端
│   └── dayjs.ts                  # 日期库配置
│
├── stores/                       # 全局状态
│   ├── useAppStore.ts
│   └── useThemeStore.ts
│
├── styles/                       # 全局样式
│   ├── globals.css
│   ├── variables.css
│   └── reset.css
│
├── types/                        # 全局类型定义
│   ├── api.ts
│   ├── common.ts
│   └── global.d.ts
│
├── utils/                        # 全局工具函数
│   ├── format.ts
│   ├── validate.ts
│   ├── storage.ts
│   └── cn.ts                     # className 合并
│
├── constants/                    # 常量
│   ├── routes.ts
│   ├── api.ts
│   └── config.ts
│
├── test/                         # 测试工具
│   ├── setup.ts
│   ├── test-utils.tsx
│   └── mocks/
│
└── index.tsx                     # 应用入口
```

### 1.2 Barrel Exports（统一导出）

```typescript
// features/auth/index.ts
export { LoginForm } from './components/LoginForm';
export { RegisterForm } from './components/RegisterForm';
export { AuthGuard } from './components/AuthGuard';
export { useAuth } from './hooks/useAuth';
export { useLogin } from './hooks/useLogin';
export type { User, AuthState } from './api/auth.types';

// 使用时
import { LoginForm, useAuth, AuthGuard } from '@/features/auth';
// 而不是
import { LoginForm } from '@/features/auth/components/LoginForm';
import { useAuth } from '@/features/auth/hooks/useAuth';
```

### 1.3 绝对路径配置

```json
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@features/*": ["src/features/*"],
      "@components/*": ["src/components/*"],
      "@hooks/*": ["src/hooks/*"],
      "@utils/*": ["src/utils/*"],
      "@lib/*": ["src/lib/*"],
      "@types/*": ["src/types/*"]
    }
  }
}
```

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

---

## 二、API 层设计

### 2.1 Axios 封装

```typescript
// src/lib/axios.ts
import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import { getToken, setToken, removeToken } from '@/utils/token';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config;

    // 401 未授权 - 尝试刷新 token
    if (error.response?.status === 401 && originalRequest) {
      try {
        const newToken = await refreshToken();
        setToken(newToken);
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch {
        removeToken();
        window.location.href = '/login';
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  }
);

// Token 刷新
let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

async function refreshToken(): Promise<string> {
  if (isRefreshing) {
    return new Promise((resolve) => {
      refreshSubscribers.push(resolve);
    });
  }

  isRefreshing = true;

  try {
    const { data } = await axios.post(
      `${import.meta.env.VITE_API_URL}/auth/refresh`,
      { refreshToken: getRefreshToken() }
    );

    refreshSubscribers.forEach((cb) => cb(data.token));
    refreshSubscribers = [];

    return data.token;
  } finally {
    isRefreshing = false;
  }
}

export default api;
```

### 2.2 API 模块化

```typescript
// src/features/products/api/products.api.ts
import api from '@/lib/axios';
import type { Product, ProductListParams, ProductListResponse } from './products.types';

export const productsApi = {
  // 获取商品列表
  getList: async (params?: ProductListParams): Promise<ProductListResponse> => {
    const { data } = await api.get('/products', { params });
    return data;
  },

  // 获取商品详情
  getById: async (id: string): Promise<Product> => {
    const { data } = await api.get(`/products/${id}`);
    return data;
  },

  // 创建商品
  create: async (product: Omit<Product, 'id'>): Promise<Product> => {
    const { data } = await api.post('/products', product);
    return data;
  },

  // 更新商品
  update: async (id: string, product: Partial<Product>): Promise<Product> => {
    const { data } = await api.put(`/products/${id}`, product);
    return data;
  },

  // 删除商品
  delete: async (id: string): Promise<void> => {
    await api.delete(`/products/${id}`);
  },

  // 上传商品图片
  uploadImage: async (file: File): Promise<{ url: string }> => {
    const formData = new FormData();
    formData.append('image', file);
    const { data } = await api.post('/products/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },
};
```

```typescript
// src/features/products/api/products.types.ts
export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  category: string;
  images: string[];
  stock: number;
  createdAt: string;
  updatedAt: string;
}

export interface ProductListParams {
  page?: number;
  limit?: number;
  category?: string;
  search?: string;
  sortBy?: 'price' | 'createdAt' | 'name';
  sortOrder?: 'asc' | 'desc';
}

export interface ProductListResponse {
  data: Product[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}
```

### 2.3 React Query 集成

```typescript
// src/lib/queryClient.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,    // 5分钟
      gcTime: 30 * 60 * 1000,      // 30分钟（原 cacheTime）
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 0,
    },
  },
});

// src/features/products/hooks/useProducts.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { productsApi } from '../api/products.api';
import type { ProductListParams } from '../api/products.types';

const PRODUCTS_KEY = 'products';

export function useProducts(params?: ProductListParams) {
  return useQuery({
    queryKey: [PRODUCTS_KEY, params],
    queryFn: () => productsApi.getList(params),
  });
}

export function useProduct(id: string) {
  return useQuery({
    queryKey: [PRODUCTS_KEY, id],
    queryFn: () => productsApi.getById(id),
    enabled: !!id,
  });
}

export function useCreateProduct() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: productsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [PRODUCTS_KEY] });
    },
  });
}

export function useUpdateProduct() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<any> }) =>
      productsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: [PRODUCTS_KEY] });
      queryClient.invalidateQueries({ queryKey: [PRODUCTS_KEY, id] });
    },
  });
}

export function useDeleteProduct() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: productsApi.delete,
    onMutate: async (id) => {
      // 乐观更新
      await queryClient.cancelQueries({ queryKey: [PRODUCTS_KEY] });
      const previous = queryClient.getQueryData([PRODUCTS_KEY]);
      queryClient.setQueryData([PRODUCTS_KEY], (old: any) =>
        old?.filter((p: any) => p.id !== id)
      );
      return { previous };
    },
    onError: (_, __, context) => {
      // 回滚
      queryClient.setQueryData([PRODUCTS_KEY], context?.previous);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: [PRODUCTS_KEY] });
    },
  });
}
```

---

## 三、认证流程

### 3.1 JWT 认证

```typescript
// src/features/auth/api/auth.api.ts
import api from '@/lib/axios';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthResponse {
  user: {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'user';
  };
  token: string;
  refreshToken: string;
}

export const authApi = {
  login: async (data: LoginRequest): Promise<AuthResponse> => {
    const response = await api.post('/auth/login', data);
    return response.data;
  },

  register: async (data: { name: string; email: string; password: string }) => {
    const response = await api.post('/auth/register', data);
    return response.data;
  },

  logout: async () => {
    await api.post('/auth/logout');
  },

  getProfile: async () => {
    const response = await api.get('/auth/profile');
    return response.data;
  },
};
```

```typescript
// src/features/auth/stores/authStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi } from '../api/auth.api';
import type { AuthResponse } from '../api/auth.types';

interface AuthState {
  user: AuthResponse['user'] | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,

      login: async (email, password) => {
        const { user, token } = await authApi.login({ email, password });
        set({ user, token, isAuthenticated: true });
      },

      logout: async () => {
        try {
          await authApi.logout();
        } finally {
          set({ user: null, token: null, isAuthenticated: false });
        }
      },

      checkAuth: async () => {
        const { token } = get();
        if (!token) {
          set({ isLoading: false });
          return;
        }

        try {
          const user = await authApi.getProfile();
          set({ user, isAuthenticated: true, isLoading: false });
        } catch {
          set({ user: null, token: null, isAuthenticated: false, isLoading: false });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
);
```

```tsx
// src/features/auth/components/AuthGuard.tsx
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';

interface Props {
  children: React.ReactNode;
  roles?: Array<'admin' | 'user'>;
}

export function AuthGuard({ children, roles }: Props) {
  const { isAuthenticated, isLoading, user } = useAuthStore();
  const location = useLocation();

  if (isLoading) {
    return <div>加载中...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (roles && user && !roles.includes(user.role)) {
    return <Navigate to="/403" replace />;
  }

  return <>{children}</>;
}
```

```tsx
// src/app/router.tsx
import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthGuard } from '@/features/auth';

const Home = lazy(() => import('@/pages/Home'));
const Login = lazy(() => import('@/pages/Login'));
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const AdminPanel = lazy(() => import('@/pages/Admin'));
const NotFound = lazy(() => import('@/pages/NotFound'));

export function AppRouter() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div>加载中...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/dashboard"
            element={
              <AuthGuard>
                <Dashboard />
              </AuthGuard>
            }
          />
          <Route
            path="/admin/*"
            element={
              <AuthGuard roles={['admin']}>
                <AdminPanel />
              </AuthGuard>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

### 3.2 Token 刷新流程

```
登录 → 获取 access_token + refresh_token
  ↓
请求 API → 携带 access_token
  ↓
token 过期 → 返回 401
  ↓
拦截器检测到 401 → 使用 refresh_token 获取新 token
  ↓
刷新成功 → 重试原请求
刷新失败 → 跳转登录页
```

---

## 四、环境配置

### 4.1 环境变量

```bash
# .env（提交到 git，公共配置）
VITE_APP_NAME=My App
VITE_APP_VERSION=1.0.0

# .env.development（开发环境）
VITE_API_URL=http://localhost:8080/api
VITE_WS_URL=ws://localhost:8080
VITE_DEBUG=true

# .env.production（生产环境）
VITE_API_URL=https://api.example.com
VITE_WS_URL=wss://api.example.com
VITE_DEBUG=false
VITE_SENTRY_DSN=https://xxx@sentry.io/123

# .env.local（本地覆盖，git 忽略）
VITE_API_URL=http://localhost:3001/api
```

```typescript
// src/types/env.d.ts
interface ImportMetaEnv {
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;
  readonly VITE_API_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_DEBUG: string;
  readonly VITE_SENTRY_DSN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// src/constants/config.ts
export const config = {
  appName: import.meta.env.VITE_APP_NAME,
  apiUrl: import.meta.env.VITE_API_URL,
  wsUrl: import.meta.env.VITE_WS_URL,
  isDev: import.meta.env.DEV,
  isProd: import.meta.env.PROD,
} as const;
```

---

## 五、CI/CD

### 5.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-type:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck

  test:
    runs-on: ubuntu-latest
    needs: lint-and-type
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run test:run -- --coverage
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
```

### 5.2 部署策略

```
┌──────────────────────────────────────────────────────────┐
│ 推荐部署流程                                              │
├──────────────────────────────────────────────────────────┤
│ 1. 开发 → 推送到 develop 分支                            │
│ 2. CI 运行 lint + typecheck + test + build               │
│ 3. PR 到 main → Code Review + CI 检查                    │
│ 4. 合并到 main → 自动部署到 staging 环境                  │
│ 5. 手动确认 → 部署到 production                          │
│                                                          │
│ Vercel/Netlify 部署：                                    │
│ - PR → 自动预览 URL (Preview Deployment)                 │
│ - main → 自动部署到生产                                   │
│                                                          │
│ Docker 部署：                                            │
│ - CI 构建 Docker 镜像                                    │
│ - 推送到容器注册表                                        │
│ - 更新 K8s / Docker Compose                              │
└──────────────────────────────────────────────────────────┘
```

---

## 六、监控

### 6.1 Sentry 错误追踪

```bash
npm install @sentry/react
```

```typescript
// src/app/main.tsx
import * as Sentry from '@sentry/react';

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  integrations: [
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration(),
  ],
  tracesSampleRate: 1.0,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
  environment: import.meta.env.MODE,
});
```

```tsx
// 错误边界集成
import { ErrorBoundary } from '@sentry/react';

function App() {
  return (
    <Sentry.ErrorBoundary fallback={<ErrorPage />}>
      <AppRouter />
    </Sentry.ErrorBoundary>
  );
}

// 手动捕获错误
import * as Sentry from '@sentry/react';

try {
  await riskyOperation();
} catch (error) {
  Sentry.captureException(error, {
    tags: { section: 'checkout' },
    extra: { orderId: '123' },
  });
}

// 添加面包屑
Sentry.addBreadcrumb({
  category: 'auth',
  message: 'User logged in',
  level: 'info',
});
```

### 6.2 Web Vitals

```typescript
// src/utils/webVitals.ts
import { onCLS, onFCP, onINP, onLCP, onTTFB } from 'web-vitals';

function sendToAnalytics(metric: any) {
  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType,
  });

  // 发送到分析服务
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/vitals', body);
  } else {
    fetch('/api/vitals', { body, method: 'POST', keepalive: true });
  }
}

onCLS(sendToAnalytics);
onFCP(sendToAnalytics);
onINP(sendToAnalytics);
onLCP(sendToAnalytics);
onTTFB(sendToAnalytics);
```

---

## 总结

| 概念 | 要点 |
|------|------|
| **目录结构** | Feature-based 组织，每个功能模块独立完整 |
| **API 层** | Axios 封装 + 拦截器 + Token 刷新 + 模块化 |
| **认证** | JWT + refresh token + AuthGuard + Zustand 持久化 |
| **环境配置** | .env 分环境，类型声明，统一 config 对象 |
| **CI/CD** | lint → test → build → deploy，PR 预览 |
| **监控** | Sentry 错误追踪 + Web Vitals 性能监控 |
