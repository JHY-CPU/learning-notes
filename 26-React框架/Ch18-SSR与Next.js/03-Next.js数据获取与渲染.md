# Next.js 数据获取与渲染

## 一、Server Components 中的数据获取

### 1.1 直接在组件中 fetch

```tsx
// app/products/page.tsx
export default async function ProductsPage() {
  // 直接在 Server Component 中使用 async/await
  const products = await fetch('https://api.example.com/products')
    .then(res => res.json());

  return (
    <div>
      <h1>商品列表</h1>
      {products.map(p => (
        <div key={p.id}>{p.name} - ¥{p.price}</div>
      ))}
    </div>
  );
}

// 优势：
// ✅ 不需要 useEffect + useState 样板代码
// ✅ 无水合不匹配问题
// ✅ 可以直接访问数据库（无需 API 层）
// ✅ 自动代码分割（大型依赖不进 bundle）
```

### 1.2 直接访问数据库

```tsx
// app/users/page.tsx
import { db } from '@/lib/database';

export default async function UsersPage() {
  // 直接查数据库！无需 API 中间层
  const users = await db.user.findMany({
    select: { id: true, name: true, email: true },
    orderBy: { createdAt: 'desc' },
  });

  return (
    <div>
      <h1>用户列表</h1>
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name} ({user.email})</li>
        ))}
      </ul>
    </div>
  );
}
```

### 1.3 并行数据获取

```tsx
// ❌ 串行 - 浪费时间（总耗时 1200ms）
export default async function Dashboard() {
  const user = await getUser();      // 400ms
  const stats = await getStats();    // 400ms
  const orders = await getOrders();  // 400ms
  return <DashboardUI user={user} stats={stats} orders={orders} />;
}

// ✅ 并行 - 总耗时 400ms（取最慢的）
export default async function Dashboard() {
  const [user, stats, orders] = await Promise.all([
    getUser(),
    getStats(),
    getOrders(),
  ]);

  return <DashboardUI user={user} stats={stats} orders={orders} />;
}

// ✅ 带错误处理的并行
export default async function Dashboard() {
  const [userResult, statsResult, ordersResult] = await Promise.allSettled([
    getUser(),
    getStats(),
    getOrders(),
  ]);

  const user = userResult.status === 'fulfilled' ? userResult.value : null;
  const stats = statsResult.status === 'fulfilled' ? statsResult.value : null;
  const orders = ordersResult.status === 'fulfilled' ? ordersResult.value : [];

  return <DashboardUI user={user} stats={stats} orders={orders} />;
}
```

---

## 二、fetch 缓存与重新验证

### 2.1 Next.js fetch 扩展

```tsx
// Next.js 扩展了原生 fetch，添加了缓存控制选项

// === 缓存行为 ===

// 1. 默认：自动缓存（同 static）
const data = await fetch('https://api.example.com/posts');

// 2. 强制缓存（等同默认）
const data = await fetch('https://api/example.com/posts', {
  cache: 'force-cache',
});

// 3. 不缓存（每次请求都获取新数据）
const data = await fetch('https://api.example.com/posts', {
  cache: 'no-store',  // 等同 SSR
});

// === 重新验证 ===

// 4. 基于时间的重新验证（ISR）
const data = await fetch('https://api.example.com/posts', {
  next: { revalidate: 60 },  // 每 60 秒重新验证
});

// 5. 按需重新验证（On-demand Revalidation）
// 配合 revalidateTag 或 revalidatePath 使用
const data = await fetch('https://api.example.com/posts', {
  next: { tags: ['posts'] },  // 打标签
});
```

### 2.2 缓存行为总结

```
┌─────────────────────────┬──────────┬──────────┬────────────────┐
│ 配置                     │ 构建时   │ 请求时   │ 重新验证       │
├─────────────────────────┼──────────┼──────────┼────────────────┤
│ (默认/force-cache)       │ ✅ 获取  │ 使用缓存 │ 永不（需手动） │
│ no-store                 │ ❌       │ ✅ 获取  │ 每次           │
│ revalidate: 60           │ ✅ 获取  │ 使用缓存 │ 60秒后         │
│ revalidate: 0            │ ❌       │ ✅ 获取  │ 每次           │
└─────────────────────────┴──────────┴──────────┴────────────────┘
```

### 2.3 非 fetch 数据源的缓存

```tsx
// 使用 unstable_cache 包装非 fetch 的数据获取
import { unstable_cache } from 'next/cache';

// 缓存数据库查询
const getCachedUser = unstable_cache(
  async (id: string) => {
    return db.user.findUnique({ where: { id } });
  },
  ['user'],                    // cache key 前缀
  { tags: ['user'], revalidate: 3600 }  // 选项
);

// 使用
export default async function UserPage({ params }) {
  const user = await getCachedUser(params.id);
  return <div>{user.name}</div>;
}
```

---

## 三、generateStaticParams (SSG)

### 3.1 基本用法

```tsx
// app/blog/[slug]/page.tsx

// 生成静态参数 - 在构建时预渲染这些页面
export async function generateStaticParams() {
  const posts = await fetch('https://api.example.com/posts')
    .then(res => res.json());

  return posts.map((post) => ({
    slug: post.slug,
  }));
}

// 页面组件
export default async function BlogPost({
  params,
}: {
  params: { slug: string };
}) {
  const post = await getPost(params.slug);
  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </article>
  );
}
```

### 3.2 动态参数控制

```tsx
// dynamicParams 配置
// app/products/[id]/page.tsx

// false: 只有 generateStaticParams 返回的参数才会被渲染，其他返回 404
export const dynamicParams = false;

export async function generateStaticParams() {
  const products = await getTopProducts();
  return products.map(p => ({ id: p.id }));
}

// true（默认）: 未预生成的参数会按需生成
export const dynamicParams = true;
```

### 3.3 Catch-all 路由的静态生成

```tsx
// app/shop/[...slug]/page.tsx
export async function generateStaticParams() {
  return [
    { slug: ['electronics'] },
    { slug: ['electronics', 'phones'] },
    { slug: ['electronics', 'laptops'] },
    { slug: ['clothing', 'shirts'] },
  ];
}

// 生成的页面：
// /shop/electronics
// /shop/electronics/phones
// /shop/electronics/laptops
// /shop/clothing/shirts
```

---

## 四、generateMetadata (SEO)

### 4.1 静态 Metadata

```tsx
// app/about/page.tsx
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: '关于我们',
  description: '了解我们的团队和使命',
  keywords: ['关于', '团队', '公司'],
  openGraph: {
    title: '关于我们 | My App',
    description: '了解我们的团队和使命',
    images: ['/og-about.png'],
  },
};

export default function AboutPage() {
  return <h1>关于我们</h1>;
}
```

### 4.2 动态 Metadata

```tsx
// app/blog/[slug]/page.tsx
import type { Metadata } from 'next';

type Props = { params: { slug: string } };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const post = await getPost(params.slug);

  if (!post) {
    return { title: '文章未找到' };
  }

  return {
    title: post.title,
    description: post.excerpt,
    authors: [{ name: post.author.name }],
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: 'article',
      publishedTime: post.publishedAt,
      modifiedTime: post.updatedAt,
      authors: [post.author.name],
      images: [
        {
          url: post.coverImage,
          width: 1200,
          height: 630,
          alt: post.title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      images: [post.coverImage],
    },
    alternates: {
      canonical: `https://example.com/blog/${params.slug}`,
    },
    robots: {
      index: true,
      follow: true,
    },
  };
}

export default async function BlogPost({ params }: Props) {
  const post = await getPost(params.slug);
  return <article>...</article>;
}
```

### 4.3 模板化 Metadata

```tsx
// app/layout.tsx
export const metadata: Metadata = {
  title: {
    default: 'My App - 最好的应用',
    template: '%s | My App',  // 子页面替换 %s
  },
};

// app/blog/page.tsx
export const metadata: Metadata = {
  title: '博客',  // 实际标题：博客 | My App
};

// app/blog/[slug]/page.tsx
export async function generateMetadata({ params }): Promise<Metadata> {
  const post = await getPost(params.slug);
  return {
    title: post.title,  // 实际标题：文章标题 | My App
  };
}
```

---

## 五、Server Actions（服务端操作）

### 5.1 基本用法

```tsx
// app/actions.ts
'use server';  // 标记为 Server Actions

export async function createPost(formData: FormData) {
  const title = formData.get('title') as string;
  const content = formData.get('content') as string;

  // 直接操作数据库
  const post = await db.post.create({
    data: { title, content },
  });

  // 可以调用 revalidate
  revalidatePath('/blog');

  return { success: true, postId: post.id };
}

export async function deletePost(id: string) {
  await db.post.delete({ where: { id } });
  revalidatePath('/blog');
}
```

### 5.2 在组件中使用

```tsx
// 方式1：表单 action
// app/blog/new/page.tsx
import { createPost } from '../actions';

export default function NewPostPage() {
  return (
    <form action={createPost}>
      <input name="title" placeholder="标题" required />
      <textarea name="content" placeholder="内容" required />
      <button type="submit">发布</button>
    </form>
  );
}

// 方式2：useActionState（React 19）
'use client';
import { useActionState } from 'react';
import { createPost } from './actions';

export function CreatePostForm() {
  const [state, formAction, isPending] = useActionState(createPost, null);

  return (
    <form action={formAction}>
      <input name="title" placeholder="标题" disabled={isPending} />
      <button type="submit" disabled={isPending}>
        {isPending ? '发布中...' : '发布'}
      </button>
      {state?.error && <p className="text-red-500">{state.error}</p>}
    </form>
  );
}

// 方式3：直接调用
'use client';
import { deletePost } from './actions';

export function DeleteButton({ postId }: { postId: string }) {
  const handleDelete = async () => {
    if (!confirm('确定删除？')) return;
    await deletePost(postId);
  };

  return (
    <button onClick={handleDelete} className="text-red-500">
      删除
    </button>
  );
}
```

### 5.3 带验证的 Server Action

```tsx
'use server';

import { z } from 'zod';

const postSchema = z.object({
  title: z.string().min(1, '标题不能为空').max(100),
  content: z.string().min(10, '内容至少10个字符'),
  categoryId: z.string().uuid(),
});

export async function createPost(prevState: any, formData: FormData) {
  const rawData = {
    title: formData.get('title'),
    content: formData.get('content'),
    categoryId: formData.get('categoryId'),
  };

  // 验证
  const result = postSchema.safeParse(rawData);

  if (!result.success) {
    return {
      errors: result.error.flatten().fieldErrors,
      message: '验证失败',
    };
  }

  try {
    await db.post.create({ data: result.data });
    revalidatePath('/blog');
    redirect('/blog');  // 重定向
  } catch (error) {
    return { message: '创建失败，请重试' };
  }
}
```

---

## 六、revalidatePath 和 revalidateTag

### 6.1 revalidatePath

```tsx
'use server';

import { revalidatePath } from 'next/cache';

export async function createPost(data: PostData) {
  await db.post.create({ data });

  // 重新验证 /blog 路径的缓存
  revalidatePath('/blog');

  // 重新验证特定动态路径
  revalidatePath('/blog/[slug]', 'page');

  // 重新验证布局及其所有子页面
  revalidatePath('/dashboard', 'layout');
}
```

### 6.2 revalidateTag

```tsx
'use server';

import { revalidateTag } from 'next/cache';

// 数据获取时打标签
async function getPosts() {
  const res = await fetch('https://api.example.com/posts', {
    next: { tags: ['posts'] },
  });
  return res.json();
}

async function getPost(slug: string) {
  const res = await fetch(`https://api.example.com/posts/${slug}`, {
    next: { tags: ['posts', `post-${slug}`] },
  });
  return res.json();
}

// 修改数据后按标签重新验证
export async function createPost(data: PostData) {
  await db.post.create({ data });
  revalidateTag('posts');  // 所有标有 'posts' 的请求都会重新验证
}

export async function updatePost(slug: string, data: PostData) {
  await db.post.update({ where: { slug }, data });
  revalidateTag('posts');
  revalidateTag(`post-${slug}`);  // 只重新验证这篇文章
}
```

---

## 七、cookies 和 headers

### 7.1 在 Server Components 中使用

```tsx
import { cookies, headers } from 'next/headers';

export default async function Dashboard() {
  // 获取 cookies
  const cookieStore = cookies();
  const theme = cookieStore.get('theme')?.value ?? 'light';
  const token = cookieStore.get('auth-token')?.value;

  // 获取 headers
  const headersList = headers();
  const userAgent = headersList.get('user-agent');
  const acceptLanguage = headersList.get('accept-language');

  // 根据 token 获取用户信息
  const user = token ? await getUserFromToken(token) : null;

  return (
    <div data-theme={theme}>
      <h1>欢迎, {user?.name ?? '访客'}</h1>
      <p>UA: {userAgent}</p>
    </div>
  );
}
```

### 7.2 设置 cookies（在 Server Actions 中）

```tsx
'use server';

import { cookies } from 'next/headers';

export async function login(formData: FormData) {
  const email = formData.get('email') as string;
  const password = formData.get('password') as string;

  const { token, user } = await authenticate(email, password);

  // 设置 cookie
  cookies().set('auth-token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7,  // 7 天
    path: '/',
  });

  return { user };
}

export async function logout() {
  cookies().delete('auth-token');
}
```

---

## 八、流式渲染与 Suspense

### 8.1 Suspense 边界

```tsx
// app/dashboard/page.tsx
import { Suspense } from 'react';

export default function DashboardPage() {
  return (
    <div>
      <h1>仪表盘</h1>

      {/* 每个区块独立加载，不阻塞其他区块 */}
      <Suspense fallback={<StatsSkeleton />}>
        <Stats />
      </Suspense>

      <Suspense fallback={<ChartSkeleton />}>
        <RevenueChart />
      </Suspense>

      <Suspense fallback={<TableSkeleton />}>
        <RecentOrders />
      </Suspense>
    </div>
  );
}

// 每个子组件独立获取数据
async function Stats() {
  const stats = await getStats();  // 500ms
  return <div>{/* stats UI */}</div>;
}

async function RevenueChart() {
  const data = await getRevenueData();  // 800ms
  return <div>{/* chart UI */}</div>;
}

async function RecentOrders() {
  const orders = await getRecentOrders();  // 300ms
  return <div>{/* orders UI */}</div>;
}
```

### 8.2 流式加载时间线

```
传统 SSR:
Server: [====生成完整HTML(1600ms)====] → 返回
Client:                               [==显示==]

流式 SSR:
Server: [==Shell==][Stats==][Chart=====][Orders=]
Client: [==Shell==][Stats==][Chart=====][Orders=]
                ↑   先显示  ↑ 继续加载  ↑ 全部完成
                50ms       500ms       800ms      1600ms

用户体验：
- 50ms: 看到页面框架和标题
- 500ms: 看到统计卡片
- 800ms: 看到图表
- 1600ms: 所有内容加载完成
```

---

## 九、错误处理与数据获取

### 9.1 try-catch 处理

```tsx
// app/products/page.tsx
export default async function ProductsPage() {
  let products;
  let error = null;

  try {
    products = await getProducts();
  } catch (err) {
    error = err instanceof Error ? err.message : '获取商品失败';
  }

  if (error) {
    return (
      <div className="text-center py-10">
        <h2 className="text-red-600">加载失败</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div>
      {products!.map(p => <ProductCard key={p.id} product={p} />)}
    </div>
  );
}
```

### 9.2 使用 error.tsx 边界

```tsx
// app/products/error.tsx
'use client';

export default function ProductsError({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <div className="text-center py-10">
      <h2>商品加载失败</h2>
      <p className="text-gray-600">{error.message}</p>
      <button onClick={reset} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
        重试
      </button>
    </div>
  );
}
```

---

## 总结

| 概念 | 要点 |
|------|------|
| **Server Component 数据获取** | 直接 async/await，不需要 useEffect |
| **fetch 缓存** | 默认缓存，no-store 获取最新，revalidate 定时刷新 |
| **generateStaticParams** | 定义构建时预渲染的动态路由参数 |
| **generateMetadata** | 动态生成 SEO 元数据 |
| **Server Actions** | 服务端函数，可直接在表单和按钮中调用 |
| **revalidate** | Path 或 Tag 级别的缓存失效 |
| **Suspense 流式** | 各区块独立加载，不相互阻塞 |
