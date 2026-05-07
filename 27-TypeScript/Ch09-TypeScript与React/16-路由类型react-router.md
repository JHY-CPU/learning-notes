# 路由类型react-router

## 一、概念说明

react-router v6+ 提供了完整的 TypeScript 支持，包括路由参数类型、loader/action 类型、Navigate 类型等。正确类型化路由可以确保参数传递和页面导航的类型安全。

## 二、具体用法

### 2.1 路由参数类型

```tsx
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';

// useParams — 动态路由参数
function UserDetail() {
  // 泛型指定参数名称和类型
  const { userId } = useParams<{ userId: string }>();

  // 如果参数可能不存在
  const { postId } = useParams<'postId'>();

  return <div>用户ID: {userId}</div>;
}

// 路由定义
// <Route path="/users/:userId" element={<UserDetail />} />
```

### 2.2 Search Params 类型

```tsx
function SearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // searchParams.get() 返回 string | null
  const query = searchParams.get('q') ?? '';
  const page = Number(searchParams.get('page') ?? '1');

  // 类型安全的参数更新
  const updateQuery = (newQuery: string) => {
    setSearchParams(prev => {
      prev.set('q', newQuery);
      prev.set('page', '1');
      return prev;
    });
  };

  return <div>搜索: {query}, 页码: {page}</div>;
}
```

### 2.3 导航类型

```tsx
import { useNavigate, Link, Navigate } from 'react-router-dom';

function Navigation() {
  const navigate = useNavigate();

  // navigate 接受 To 类型
  const goToUser = (id: number) => {
    navigate(`/users/${id}`);           // 字符串路径
    navigate({ pathname: '/users', search: `?id=${id}` }); // 对象形式
  };

  return (
    <div>
      {/* Link 组件 — 自动类型检查 to 属性 */}
      <Link to="/about">关于</Link>
      <Link to={{ pathname: '/users', search: '?sort=name' }}>用户列表</Link>

      {/* 带状态的导航 */}
      <Link to="/dashboard" state={{ from: 'home' }}>
        仪表盘
      </Link>
    </div>
  );
}
```

### 2.4 Loader / Action 类型

```tsx
import { LoaderFunctionArgs, ActionFunctionArgs } from 'react-router-dom';

// Loader — 加载路由数据
interface UserData {
  id: number;
  name: string;
}

export async function userLoader({ params }: LoaderFunctionArgs): Promise<UserData> {
  const response = await fetch(`/api/users/${params.userId}`);
  if (!response.ok) throw new Response('Not Found', { status: 404 });
  return response.json();
}

// Action — 处理表单提交
export async function userAction({ request, params }: ActionFunctionArgs) {
  const formData = await request.formData();
  const name = formData.get('name') as string;

  await fetch(`/api/users/${params.userId}`, {
    method: 'PUT',
    body: JSON.stringify({ name }),
  });

  return { success: true };
}

// 在组件中使用 loader 数据
import { useLoaderData } from 'react-router-dom';

function UserPage() {
  const user = useLoaderData() as UserData;
  return <div>{user.name}</div>;
}
```

### 2.5 路由配置类型

```tsx
import { RouteObject } from 'react-router-dom';

const routes: RouteObject[] = [
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      {
        path: 'users/:userId',
        element: <UserDetail />,
        loader: userLoader,
        action: userAction,
      },
      { path: '*', element: <NotFound /> },
    ],
  },
];
```

## 三、注意事项与常见陷阱

1. **`useParams` 泛型应匹配路由定义的参数**：确保参数名称一致
2. **`searchParams.get()` 返回 `string | null`**：始终做 null 检查
3. **loader 返回值需要手动断言**：`useLoaderData()` 返回 `unknown`
4. **路由配置对象的类型用 `RouteObject`**：确保配置正确
5. **React Router v7 将有更大的类型改进**：关注版本更新
