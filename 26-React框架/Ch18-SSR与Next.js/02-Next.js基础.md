# Next.js 基础

## 一、App Router 概览

### 1.1 项目结构

```
my-app/
├── app/                    # App Router（Next.js 13+）
│   ├── layout.tsx          # 根布局
│   ├── page.tsx            # 首页
│   ├── loading.tsx         # 加载状态
│   ├── error.tsx           # 错误边界
│   ├── not-found.tsx       # 404 页面
│   ├── globals.css         # 全局样式
│   ├── about/
│   │   └── page.tsx        # /about
│   ├── blog/
│   │   ├── layout.tsx      # /blog 布局
│   │   ├── page.tsx        # /blog
│   │   └── [slug]/
│   │       └── page.tsx    # /blog/:slug
│   └── api/
│       └── route.ts        # API 路由
├── public/                 # 静态资源
├── next.config.js          # 配置文件
├── tailwind.config.js      # Tailwind 配置
├── tsconfig.json
└── package.json
```

### 1.2 App Router vs Pages Router

```
┌──────────────┬───────────────────────┬────────────────────────┐
│              │ Pages Router          │ App Router             │
├──────────────┼───────────────────────┼────────────────────────┤
│ 目录         │ pages/                │ app/                   │
│ 路由         │ 文件名即路由          │ 文件夹+page.tsx        │
│ 数据获取     │ getServerSideProps    │ async Server Components│
│              │ getStaticProps        │                        │
│ 布局         │ 手动实现              │ layout.tsx 原生支持    │
│ 组件类型     │ 默认客户端            │ 默认服务端             │
│ 流式渲染     │ 不支持                │ Suspense 原生支持      │
│ 推荐状态     │ 生产环境可用          │ 推荐使用               │
└──────────────┴───────────────────────┴────────────────────────┘
```

---

## 二、文件路由系统

### 2.1 路由定义

```
app/page.tsx                    → /
app/about/page.tsx              → /about
app/blog/page.tsx               → /blog
app/blog/[slug]/page.tsx        → /blog/:slug
app/blog/[slug]/comments/page.tsx → /blog/:slug/comments
app/shop/[...slug]/page.tsx     → /shop/*（catch-all）
app/shop/[[...slug]]/page.tsx   → /shop/* 或 /shop（可选 catch-all）
```

```tsx
// app/blog/[slug]/page.tsx
interface Props {
  params: { slug: string };
}

export default async function BlogPost({ params }: Props) {
  const post = await getPost(params.slug);
  return (
    <article>
      <h1>{post.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  );
}

// 生成静态参数（用于 SSG）
export async function generateStaticParams() {
  const posts = await getAllPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

// 动态路由类型（可选）
export const dynamicParams = false;  // 未在 generateStaticParams 中的返回 404
// export const dynamicParams = true;  // 默认，未生成的按需渲染
```

### 2.2 路由组 (Route Groups)

```
app/
├── (marketing)/              # 路由组 - 不影响 URL
│   ├── layout.tsx            # 营销页面布局
│   ├── page.tsx              → /
│   ├── about/page.tsx        → /about
│   └── pricing/page.tsx      → /pricing
├── (shop)/                   # 路由组
│   ├── layout.tsx            # 商店布局（有侧边栏）
│   ├── products/page.tsx     → /products
│   └── cart/page.tsx         → /cart
└── (auth)/
    ├── layout.tsx            # 认证布局（居中卡片）
    ├── login/page.tsx        → /login
    └── register/page.tsx     → /register
```

---

## 三、layout.tsx 布局系统

### 3.1 根布局

```tsx
// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: {
    template: '%s | My App',  // 子页面可以设置 %s 部分
    default: 'My App',
  },
  description: '一个使用 Next.js 构建的应用',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className={inter.className}>
        <Navbar />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
```

### 3.2 嵌套布局

```tsx
// app/dashboard/layout.tsx
import { Sidebar } from '@/components/Sidebar';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1">{children}</div>
    </div>
  );
}

// dashboard 下所有页面都有 Sidebar
// /dashboard          → Sidebar + DashboardPage
// /dashboard/settings → Sidebar + SettingsPage
// /dashboard/users    → Sidebar + UsersPage

// 重要特性：布局在导航时不会重新渲染（保持状态！）
// 比如 Sidebar 中的展开状态在页面切换时保持
```

### 3.3 布局的状态保持

```
用户在 /dashboard 页面展开了 Sidebar 的子菜单
然后点击导航到 /dashboard/settings

传统路由：整个页面重新渲染 → Sidebar 状态丢失
Next.js 布局：Layout 组件不重新渲染 → Sidebar 状态保持！

这就是为什么布局适合放：
- 导航栏
- 侧边栏
- 用户状态信息
- 需要跨页面保持的 UI
```

---

## 四、page.tsx 页面

### 4.1 Server Component 页面（默认）

```tsx
// app/products/page.tsx
import { ProductCard } from '@/components/ProductCard';

// 这是 Server Component - 在服务端执行
export default async function ProductsPage() {
  // 可以直接使用 async/await
  const products = await fetch('https://api.example.com/products', {
    cache: 'no-store',  // 每次请求都获取新数据
  }).then(res => res.json());

  return (
    <div>
      <h1>所有商品</h1>
      <div className="grid grid-cols-3 gap-4">
        {products.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
}
```

### 4.2 Client Component 页面

```tsx
// app/interactive/page.tsx
'use client';  // 标记为 Client Component

import { useState } from 'react';

export default function InteractivePage() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}

// 注意：'use client' 会将该组件及其所有子组件都变为客户端组件
// 如果子组件不需要交互，应该把它提取出去作为 Server Component
```

---

## 五、loading.tsx 加载状态

### 5.1 基本用法

```tsx
// app/dashboard/loading.tsx
export default function DashboardLoading() {
  return (
    <div className="animate-pulse">
      <div className="h-8 bg-gray-200 rounded w-1/3 mb-4" />
      <div className="space-y-3">
        <div className="h-4 bg-gray-200 rounded" />
        <div className="h-4 bg-gray-200 rounded w-5/6" />
        <div className="h-4 bg-gray-200 rounded w-4/6" />
      </div>
    </div>
  );
}
```

### 5.2 工作原理

```
用户访问 /dashboard
  ↓
Next.js 立即显示 loading.tsx（instant loading state）
  ↓
同时在服务端执行 page.tsx
  ↓
page.tsx 准备完成后替换 loading.tsx

类似 Suspense 的 fallback，但更简便：
// 不需要手动写 Suspense
<Suspense fallback={<Loading />}>
  <Dashboard />
</Suspense>
// Next.js 自动处理
```

### 5.3 流式加载

```tsx
// app/dashboard/loading.tsx - 骨架屏
export default function Loading() {
  return (
    <div className="p-6">
      <div className="animate-pulse space-y-4">
        <div className="h-10 bg-gray-200 rounded w-48" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg" />
          ))}
        </div>
      </div>
    </div>
  );
}

// app/dashboard/page.tsx - 实际内容
export default async function Dashboard() {
  const [stats, recentOrders] = await Promise.all([
    getStats(),        // 可能需要 500ms
    getRecentOrders(), // 可能需要 800ms
  ]);

  return (
    <div className="p-6">
      <h1>仪表盘</h1>
      <div className="grid grid-cols-3 gap-4">
        <StatCard title="总用户" value={stats.users} />
        <StatCard title="总收入" value={stats.revenue} />
        <StatCard title="订单数" value={stats.orders} />
      </div>
      <RecentOrdersTable orders={recentOrders} />
    </div>
  );
}
```

---

## 六、error.tsx 错误处理

### 6.1 基本用法

```tsx
// app/dashboard/error.tsx
'use client';  // 必须是 Client Component

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="p-6 text-center">
      <h2 className="text-xl font-bold text-red-600">出了点问题</h2>
      <p className="text-gray-600 mt-2">{error.message}</p>
      <button
        onClick={reset}  // 尝试重新渲染
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        重试
      </button>
    </div>
  );
}
```

### 6.2 全局错误页面

```tsx
// app/global-error.tsx（只能用于根 layout）
'use client';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <html>
      <body>
        <div className="min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h1>应用出错了</h1>
            <button onClick={reset}>重新加载</button>
          </div>
        </div>
      </body>
    </html>
  );
}
```

### 6.3 not-found.tsx

```tsx
// app/not-found.tsx
import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center">
      <h1 className="text-6xl font-bold text-gray-300">404</h1>
      <p className="text-xl text-gray-600 mt-4">页面不存在</p>
      <Link
        href="/"
        className="mt-6 px-6 py-3 bg-blue-500 text-white rounded-lg"
      >
        返回首页
      </Link>
    </div>
  );
}

// 在代码中触发 404
import { notFound } from 'next/navigation';

async function getProduct(id: string) {
  const product = await db.product.findUnique({ where: { id } });
  if (!product) notFound();  // 渲染 not-found.tsx
  return product;
}
```

---

## 七、Server Components vs Client Components

### 7.1 核心区别

```
Server Components (默认):
✅ 可以直接访问数据库
✅ 可以使用文件系统
✅ 可以使用服务端密钥
✅ 零客户端 JS 开销
✅ 自动代码分割
❌ 不能使用 useState/useEffect
❌ 不能使用浏览器 API
❌ 不能有事件处理器

Client Components ('use client'):
✅ 可以使用所有 React hooks
✅ 可以使用浏览器 API
✅ 可以有事件处理器
✅ 可以使用状态
❌ 不能直接访问数据库
❌ 会增加客户端 bundle 大小
```

### 7.2 使用原则

```
默认用 Server Component，只在需要时添加 'use client'

需要 'use client' 的场景：
- 使用 useState、useEffect、useReducer 等 hooks
- 使用浏览器 API（window、document、localStorage）
- 添加事件处理器（onClick、onChange）
- 使用 useEffect 的生命周期效果
- 使用自定义 hooks（这些 hooks 内部使用了上述功能）
- 使用 context provider

仍然是 Server Component 的场景：
- 从数据库/API 获取数据并展示
- 纯展示组件（不交互）
- 在服务端保留敏感信息（密钥、令牌）
- 大型依赖（如 markdown 解析器）- 不会发送到客户端
```

### 7.3 组合模式

```tsx
// ✅ Server Component 包含 Client Component
// app/page.tsx (Server Component)
import { LikeButton } from './LikeButton';  // Client Component

export default async function Page() {
  const post = await getPost(1);

  return (
    <article>
      <h1>{post.title}</h1>           {/* 服务端渲染 */}
      <p>{post.content}</p>            {/* 服务端渲染 */}
      <LikeButton postId={post.id} />  {/* 客户端交互 */}
    </article>
  );
}

// components/LikeButton.tsx (Client Component)
'use client';

import { useState } from 'react';

export function LikeButton({ postId }: { postId: string }) {
  const [liked, setLiked] = useState(false);

  return (
    <button onClick={() => setLiked(!liked)}>
      {liked ? '已点赞' : '点赞'}
    </button>
  );
}

// ✅ Server Component 作为 children 传给 Client Component
// components/Tabs.tsx (Client Component)
'use client';

import { useState } from 'react';

export function Tabs({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState(0);
  // children 是 Server Component，不会变成客户端代码
  return <div>{children}</div>;
}

// app/page.tsx (Server Component)
import { Tabs } from './Tabs';
import { ExpensiveComponent } from './ExpensiveComponent';

export default function Page() {
  return (
    <Tabs>
      {/* ExpensiveComponent 是 Server Component，其代码不会发送到客户端 */}
      <ExpensiveComponent />
    </Tabs>
  );
}
```

---

## 八、Link 组件

### 8.1 基本用法

```tsx
import Link from 'next/link';

// 基本导航
<Link href="/about">关于我们</Link>

// 动态路由
<Link href={`/blog/${post.slug}`}>{post.title}</Link>

// 带查询参数
<Link href={{ pathname: '/search', query: { q: 'react' } }}>
  搜索 React
</Link>

// 预加载控制
<Link href="/dashboard" prefetch={false}>  {/* 禁用预加载 */}
  仪表盘
</Link>

// 替换历史记录（不会添加到历史栈）
<Link href="/new-page" replace>
  跳转（不可返回）
</Link>

// 外部链接会自动变为 <a> 标签
<Link href="https://github.com">GitHub</Link>

// 样式 - 使用 className 或 style
<Link href="/about" className="text-blue-500 hover:underline">
  关于
</Link>
```

### 8.2 编程式导航

```tsx
'use client';

import { useRouter } from 'next/navigation';

export function SearchForm() {
  const router = useRouter();

  const handleSubmit = (query: string) => {
    router.push(`/search?q=${encodeURIComponent(query)}`);
  };

  const handleBack = () => {
    router.back();
  };

  const handleReplace = () => {
    router.replace('/dashboard');  // 替换当前历史记录
  };

  const handleRefresh = () => {
    router.refresh();  // 刷新当前页面（重新获取服务端数据）
  };

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      handleSubmit(new FormData(e.currentTarget).get('q') as string);
    }}>
      <input name="q" />
      <button type="submit">搜索</button>
    </form>
  );
}
```

### 8.3 路由拦截 (Intercepting Routes)

```
app/
├── feed/
│   └── page.tsx              → /feed（图片列表）
├── photo/
│   └── [id]/
│       └── page.tsx          → /photo/:id（全屏照片页）
└── @modal/                   # 拦截路由
    ├── default.tsx
    └── (.)photo/             # 拦截 /photo 路由
        └── [id]/
            └── page.tsx      → 在 modal 中显示照片

(.) 表示拦截同一层级的路由
(..) 表示拦截上一层级
(...) 表示拦截根目录
```

---

## 九、Next.js 配置

### 9.1 next.config.js

```tsx
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  // 图片优化
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'example.com',
      },
    ],
  },

  // 重定向
  async redirects() {
    return [
      {
        source: '/old-blog/:slug',
        destination: '/blog/:slug',
        permanent: true,  // 301 重定向
      },
    ];
  },

  // 重写（URL 不变，内容来自目标）
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://api.example.com/:path*',
      },
    ];
  },

  // 自定义 headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
        ],
      },
    ];
  },

  // 实验性功能
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
};

module.exports = nextConfig;
```

### 9.2 环境变量

```bash
# .env.local（git 忽略）
DATABASE_URL=postgresql://...
SECRET_KEY=my-secret

# .env（提交到 git）
NEXT_PUBLIC_API_URL=https://api.example.com
```

```tsx
// 使用环境变量
// 服务端组件可以访问所有变量
const dbUrl = process.env.DATABASE_URL;

// 客户端只能访问 NEXT_PUBLIC_ 前缀的变量
const apiUrl = process.env.NEXT_PUBLIC_API_URL;
```

---

## 十、创建 Next.js 项目

### 10.1 快速开始

```bash
# 创建项目
npx create-next-app@latest my-app
# 选项：TypeScript, ESLint, Tailwind, src/, App Router, import alias

# 进入目录
cd my-app

# 开发
npm run dev        # http://localhost:3000

# 构建
npm run build

# 生产运行
npm start
```

### 10.2 项目依赖

```json
{
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "typescript": "^5",
    "tailwindcss": "^3",
    "postcss": "^8",
    "eslint": "^8",
    "eslint-config-next": "^14"
  }
}
```

---

## 总结

| 概念 | 要点 |
|------|------|
| **App Router** | 基于文件系统的路由，目录即路由 |
| **layout.tsx** | 嵌套布局，导航时不重新渲染，保持状态 |
| **page.tsx** | 路由页面，支持 async 直接获取数据 |
| **loading.tsx** | 类似 Suspense fallback，流式加载状态 |
| **error.tsx** | 错误边界，必须是 Client Component |
| **Server vs Client** | 默认 Server，需要交互时加 'use client' |
| **Link** | 带预加载的客户端导航组件 |
