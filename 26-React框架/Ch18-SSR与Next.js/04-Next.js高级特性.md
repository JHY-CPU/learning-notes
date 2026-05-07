# Next.js 高级特性

## 一、API Routes (route.ts)

### 1.1 基本用法

```tsx
// app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';

// GET /api/users
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get('page') ?? '1');
  const limit = parseInt(searchParams.get('limit') ?? '10');

  const users = await db.user.findMany({
    skip: (page - 1) * limit,
    take: limit,
  });

  return NextResponse.json({ users, page, limit });
}

// POST /api/users
export async function POST(request: NextRequest) {
  const body = await request.json();

  const user = await db.user.create({
    data: {
      name: body.name,
      email: body.email,
    },
  });

  return NextResponse.json(user, { status: 201 });
}
```

### 1.2 动态路由 API

```tsx
// app/api/users/[id]/route.ts
import { NextRequest, NextResponse } from 'next/server';

type Props = { params: { id: string } };

// GET /api/users/:id
export async function GET(request: NextRequest, { params }: Props) {
  const user = await db.user.findUnique({ where: { id: params.id } });

  if (!user) {
    return NextResponse.json(
      { error: '用户不存在' },
      { status: 404 }
    );
  }

  return NextResponse.json(user);
}

// PUT /api/users/:id
export async function PUT(request: NextRequest, { params }: Props) {
  const body = await request.json();

  const user = await db.user.update({
    where: { id: params.id },
    data: body,
  });

  return NextResponse.json(user);
}

// DELETE /api/users/:id
export async function DELETE(request: NextRequest, { params }: Props) {
  await db.user.delete({ where: { id: params.id } });

  return NextResponse.json({ message: '已删除' });
}
```

### 1.3 中间件式处理

```tsx
// app/api/protected/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';

export async function GET(request: NextRequest) {
  // 鉴权
  const token = await getToken({ req: request });
  if (!token) {
    return NextResponse.json({ error: '未授权' }, { status: 401 });
  }

  // 业务逻辑
  const data = await getProtectedData(token.sub);
  return NextResponse.json(data);
}

// CORS 处理
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
```

---

## 二、Middleware

### 2.1 基本用法

```tsx
// middleware.ts（项目根目录，不是 app 目录）
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // 获取 token
  const token = request.cookies.get('auth-token')?.value;

  // 需要认证的路径
  const protectedPaths = ['/dashboard', '/profile', '/settings'];
  const isProtected = protectedPaths.some(path =>
    request.nextUrl.pathname.startsWith(path)
  );

  if (isProtected && !token) {
    // 重定向到登录页
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('from', request.nextUrl.pathname);
    return NextResponse.redirect(loginUrl);
  }

  // 已登录用户访问登录页，重定向到首页
  if (request.nextUrl.pathname === '/login' && token) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  // 添加响应头
  const response = NextResponse.next();
  response.headers.set('x-pathname', request.nextUrl.pathname);

  return response;
}

// 配置匹配路径
export const config = {
  matcher: [
    /*
     * 匹配所有路径，除了：
     * - api 路由
     * - 静态文件
     * - 图片优化
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
```

### 2.2 地理位置与 A/B 测试

```tsx
// middleware.ts
export function middleware(request: NextRequest) {
  // 获取地理位置（Vercel 提供）
  const country = request.geo?.country ?? 'US';
  const city = request.geo?.city ?? 'Unknown';

  // 重定向到地区版本
  if (country === 'CN' && !request.nextUrl.pathname.startsWith('/zh')) {
    return NextResponse.redirect(
      new URL(`/zh${request.nextUrl.pathname}`, request.url)
    );
  }

  // A/B 测试
  const response = NextResponse.next();
  const bucket = Math.random() < 0.5 ? 'control' : 'variant';
  response.cookies.set('ab-bucket', bucket);

  return response;
}
```

### 2.3 限流

```tsx
// middleware.ts
import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

const ratelimit = new Ratelimit({
  redis: Redis.fromEnv(),
  limiter: Ratelimit.slidingWindow(10, '10 s'),  // 10秒内最多10次
});

export async function middleware(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith('/api/')) {
    const ip = request.ip ?? '127.0.0.1';
    const { success, limit, remaining } = await ratelimit.limit(ip);

    if (!success) {
      return NextResponse.json(
        { error: '请求过于频繁' },
        {
          status: 429,
          headers: {
            'X-RateLimit-Limit': limit.toString(),
            'X-RateLimit-Remaining': remaining.toString(),
          },
        }
      );
    }
  }

  return NextResponse.next();
}
```

---

## 三、Parallel Routes（平行路由）

### 3.1 概念

```
平行路由允许在同一布局中同时渲染多个页面

app/
├── layout.tsx              # 包含多个 slot
├── @analytics/
│   ├── page.tsx            # analytics slot 的默认内容
│   └── loading.tsx
├── @team/
│   ├── page.tsx            # team slot 的默认内容
│   └── [id]/page.tsx
└── page.tsx                # 主页面内容

URL: /dashboard
渲染: layout 中的 @analytics/page.tsx + @team/page.tsx + page.tsx
```

### 3.2 实现

```tsx
// app/layout.tsx
export default function DashboardLayout({
  children,
  analytics,
  team,
}: {
  children: React.ReactNode;
  analytics: React.ReactNode;
  team: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-3 gap-4">
      <main className="col-span-2">{children}</main>
      <aside className="space-y-4">
        {analytics}   {/* @analytics slot */}
        {team}        {/* @team slot */}
      </aside>
    </div>
  );
}

// app/@analytics/page.tsx
export default async function AnalyticsPage() {
  const stats = await getAnalytics();
  return (
    <div className="bg-white rounded-lg p-4">
      <h3>分析数据</h3>
      <div>访问量: {stats.pageViews}</div>
    </div>
  );
}

// app/@team/page.tsx
export default async function TeamPage() {
  const members = await getTeamMembers();
  return (
    <div className="bg-white rounded-lg p-4">
      <h3>团队成员</h3>
      {members.map(m => <div key={m.id}>{m.name}</div>)}
    </div>
  );
}
```

### 3.3 default.tsx（默认 slot）

```tsx
// app/@analytics/default.tsx
// 当没有匹配的子路由时显示
export default function DefaultAnalytics() {
  return <div>选择日期范围查看分析数据</div>;
}
```

---

## 四、Intercepting Routes（拦截路由）

### 4.1 场景

```
场景：在 feed 页面点击图片，希望在 modal 中显示详情，而非跳转新页面

目录结构：
app/
├── feed/
│   └── page.tsx                 → /feed（图片列表）
├── photo/
│   └── [id]/
│       └── page.tsx             → /photo/:id（全屏照片页）
└── @modal/
    ├── default.tsx
    └── (.)photo/                # 拦截 /photo 路由
        └── [id]/
            └── page.tsx         → 在 modal 中显示照片

导航行为：
- 从 /feed 点击图片 → modal 中显示（(.) 拦截）
- 直接访问 /photo/123 → 全屏显示
- 刷新 modal 页面 → 全屏显示（因为拦截只在客户端导航时生效）
```

### 4.2 实现

```tsx
// app/layout.tsx
export default function RootLayout({ children, modal }) {
  return (
    <html>
      <body>
        {children}
        {modal}  {/* @modal slot */}
      </body>
    </html>
  );
}

// app/@modal/(.)photo/[id]/page.tsx
import { Modal } from '@/components/Modal';
import { PhotoDetail } from '@/components/PhotoDetail';

export default async function InterceptedPhotoPage({
  params,
}: {
  params: { id: string };
}) {
  const photo = await getPhoto(params.id);

  return (
    <Modal>
      <PhotoDetail photo={photo} />
    </Modal>
  );
}

// app/@modal/default.tsx
export default function Default() {
  return null;  // 默认不显示 modal
}
```

---

## 五、Route Groups

### 5.1 不同布局的分组

```
app/
├── (marketing)/
│   ├── layout.tsx       # 宽屏布局，无侧边栏
│   ├── page.tsx         → /
│   ├── about/page.tsx   → /about
│   └── pricing/page.tsx → /pricing
├── (shop)/
│   ├── layout.tsx       # 商店布局，有侧边栏和购物车
│   ├── products/page.tsx → /products
│   └── cart/page.tsx    → /cart
└── (auth)/
    ├── layout.tsx       # 居中卡片布局
    ├── login/page.tsx   → /login
    └── register/page.tsx → /register
```

```tsx
// app/(auth)/layout.tsx
export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
        {children}
      </div>
    </div>
  );
}

// app/(shop)/layout.tsx
export default function ShopLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex">
      <ShopSidebar />
      <main className="flex-1 p-6">{children}</main>
      <ShoppingCart />
    </div>
  );
}
```

---

## 六、国际化 (i18n)

### 6.1 路由级 i18n

```tsx
// next.config.js
module.exports = {
  i18n: {
    locales: ['zh-CN', 'en', 'ja'],
    defaultLocale: 'zh-CN',
    localeDetection: true,  // 自动检测浏览器语言
  },
};

// 注意：i18n 配置仅在 Pages Router 中有效
// App Router 需要手动实现
```

### 6.2 App Router 中的 i18n

```
app/
├── [lang]/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── about/page.tsx
│   └── blog/page.tsx
├── dictionaries/
│   ├── zh-CN.json
│   ├── en.json
│   └── ja.json
└── middleware.ts
```

```tsx
// middleware.ts
import { NextResponse } from 'next/server';
import { match } from '@formatjs/intl-localematcher';
import Negotiator from 'negotiator';

const locales = ['zh-CN', 'en', 'ja'];
const defaultLocale = 'zh-CN';

function getLocale(request: NextRequest): string {
  const headers = { 'accept-language': request.headers.get('accept-language') ?? '' };
  const languages = new Negotiator({ headers }).languages();
  return match(languages, locales, defaultLocale);
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 检查路径是否已有语言前缀
  const pathnameHasLocale = locales.some(
    locale => pathname.startsWith(`/${locale}/`) || pathname === `/${locale}`
  );

  if (pathnameHasLocale) return;

  // 重定向到带语言前缀的路径
  const locale = getLocale(request);
  request.nextUrl.pathname = `/${locale}${pathname}`;
  return NextResponse.redirect(request.nextUrl);
}

export const config = {
  matcher: ['/((?!api|_next|.*\\..*).*)'],
};
```

```tsx
// dictionaries/zh-CN.json
{
  "home": {
    "title": "欢迎来到我的网站",
    "description": "使用 Next.js 构建的全栈应用"
  },
  "nav": {
    "about": "关于我们",
    "blog": "博客",
    "contact": "联系我们"
  }
}

// lib/dictionaries.ts
const dictionaries = {
  'zh-CN': () => import('@/dictionaries/zh-CN.json').then(m => m.default),
  'en': () => import('@/dictionaries/en.json').then(m => m.default),
  'ja': () => import('@/dictionaries/ja.json').then(m => m.default),
};

export const getDictionary = (locale: string) =>
  dictionaries[locale]?.() ?? dictionaries['zh-CN']();

// app/[lang]/page.tsx
export default async function Home({
  params: { lang },
}: {
  params: { lang: string };
}) {
  const dict = await getDictionary(lang);

  return (
    <div>
      <h1>{dict.home.title}</h1>
      <p>{dict.home.description}</p>
    </div>
  );
}
```

---

## 七、Image 优化 (next/image)

### 7.1 基本用法

```tsx
import Image from 'next/image';

// 本地图片（自动获取尺寸）
import heroImage from '@/public/hero.jpg';
<Image src={heroImage} alt="Hero" />

// 远程图片（需要配置域名）
<Image
  src="https://example.com/photo.jpg"
  alt="Photo"
  width={800}
  height={600}
  priority            // LCP 图片，预加载
  placeholder="blur"  // 模糊占位
  blurDataURL="data:image/..."  // 内联模糊图
/>

// 响应式图片
<Image
  src="/hero.jpg"
  alt="Hero"
  width={1200}
  height={600}
  sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
  className="w-full h-auto"
/>

// 填充模式（父容器有宽高）
<div className="relative w-full h-[400px]">
  <Image
    src="/hero.jpg"
    alt="Hero"
    fill
    className="object-cover"
    sizes="100vw"
  />
</div>
```

### 7.2 配置远程域名

```tsx
// next.config.js
module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
      },
      {
        protocol: 'https',
        hostname: '*.amazonaws.com',  // 支持通配符
      },
    ],
    // 图片格式优先级
    formats: ['image/avif', 'image/webp'],
    // 自定义断点
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
};
```

### 7.3 Image 优化的好处

```
next/image 自动处理：
✅ WebP/AVIF 格式转换（更小体积）
✅ 按需调整尺寸（响应式 srcset）
✅ 懒加载（非 LCP 图片）
✅ 防止布局偏移（CLS）
✅ 图片渐进加载
✅ 自动优化（压缩质量）
```

---

## 八、Font 优化 (next/font)

### 8.1 Google Fonts

```tsx
// app/layout.tsx
import { Inter, Noto_Sans_SC, JetBrains_Mono } from 'next/font/google';

// 主字体
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',  // CSS 变量
});

// 中文字体
const notoSansSC = Noto_Sans_SC({
  subsets: ['latin'],
  weight: ['400', '500', '700'],
  display: 'swap',
  variable: '--font-noto',
});

// 等宽字体
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
});

export default function RootLayout({ children }) {
  return (
    <html
      lang="zh-CN"
      className={`${inter.variable} ${notoSansSC.variable} ${jetbrainsMono.variable}`}
    >
      <body className={inter.className}>{children}</body>
    </html>
  );
}

// 使用
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-noto)', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
    },
  },
};
```

### 8.2 本地字体

```tsx
import localFont from 'next/font/local';

const myFont = localFont({
  src: [
    {
      path: '../fonts/MyFont-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
    {
      path: '../fonts/MyFont-Bold.woff2',
      weight: '700',
      style: 'normal',
    },
  ],
  variable: '--font-custom',
});

export default function RootLayout({ children }) {
  return (
    <html className={myFont.variable}>
      <body>{children}</body>
    </html>
  );
}
```

### 8.3 next/font 优化原理

```
传统方式的问题：
- @import Google Fonts → 额外网络请求 → FOUT（Flash of Unstyled Text）
- font-display: swap → 文字先用系统字体 → 字体加载后切换 → 布局偏移

next/font 的优势：
- 字体文件在构建时下载到本地
- 自动 self-hosted → 无外部请求
- 使用 size-adjust 和 ascent-override → 零布局偏移
- 自动 subset → 更小文件
- CSS 内联 → 无阻塞加载
```

---

## 九、部署

### 9.1 Vercel 部署（推荐）

```bash
# 安装 Vercel CLI
npm install -g vercel

# 部署
vercel

# 生产部署
vercel --prod

# 环境变量设置
vercel env add DATABASE_URL
vercel env add NEXT_PUBLIC_API_URL
```

```tsx
// vercel.json（通常不需要，Next.js 自动检测）
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "regions": ["hkg1"],  // 香港区域
  "functions": {
    "app/api/**/*": {
      "maxDuration": 30
    }
  }
}
```

### 9.2 Docker 部署

```dockerfile
# Dockerfile
FROM node:20-alpine AS base

# 依赖安装阶段
FROM base AS deps
WORKDIR /app
COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i --frozen-lockfile; \
  else echo "Lockfile not found." && exit 1; \
  fi

# 构建阶段
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# 生产运行阶段
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# 自动输出 standalone
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

```javascript
// next.config.js - 启用 standalone 输出
module.exports = {
  output: 'standalone',
};
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - '3000:3000'
    environment:
      - DATABASE_URL=postgresql://...
      - NEXT_PUBLIC_API_URL=https://api.example.com
    restart: unless-stopped
```

### 9.3 Nginx 反向代理

```nginx
# /etc/nginx/sites-available/myapp
server {
    listen 80;
    server_name example.com;

    # 静态文件缓存
    location /_next/static/ {
        alias /app/.next/static/;
        expires 365d;
        access_log off;
        add_header Cache-Control "public, immutable";
    }

    # 图片缓存
    location /_next/image {
        proxy_pass http://localhost:3000;
        expires 30d;
        add_header Cache-Control "public";
    }

    # API 不缓存
    location /api/ {
        proxy_pass http://localhost:3000;
        add_header Cache-Control "no-store";
    }

    # 其他请求
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml;
}
```

### 9.4 部署选项对比

```
┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│             │ Vercel   │ Docker   │ Netlify  │ AWS      │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ 配置难度    │ 零配置   │ 中等     │ 简单     │ 复杂     │
│ 全栈支持    │ ✅       │ ✅       │ ✅       │ ✅       │
│ Edge函数    │ ✅       │ ❌       │ ✅       │ ✅       │
│ 流式SSR     │ ✅       │ ✅       │ 部分     │ ✅       │
│ 图片优化    │ ✅       │ 自建     │ ✅       │ 自建     │
│ 免费额度    │ 有       │ 无       │ 有       │ 无       │
│ 适合        │ 快速上线 │ 企业部署 │ 中小项目 │ 大型项目 │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 总结

| 概念 | 要点 |
|------|------|
| **API Routes** | app/api/route.ts，支持 GET/POST/PUT/DELETE |
| **Middleware** | 请求级别拦截，用于鉴权、重定向、限流 |
| **Parallel Routes** | 同一布局中渲染多个页面（@slot） |
| **Intercepting Routes** | 在当前页面显示目标路由内容（(.)语法） |
| **i18n** | App Router 需手动实现，配合 middleware 重定向 |
| **next/image** | 自动优化图片格式、尺寸、懒加载 |
| **next/font** | 自托管字体，零布局偏移 |
| **部署** | Vercel 零配置，Docker 适合企业自建 |
