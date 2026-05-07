# React Hooks

## 一、useQuery

```typescript
import { useQuery, gql } from '@apollo/client';

const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      orders(first: 5) {
        edges {
          node { id total status }
        }
      }
    }
  }
`;

function UserProfile({ userId }) {
  const { data, loading, error, refetch, fetchMore } = useQuery(GET_USER, {
    variables: { id: userId },
    pollInterval: 30000,           // 轮询
    skip: !userId,                 // 条件执行
    notifyOnNetworkStatusChange: true,
  });

  if (loading) return <Skeleton />;
  if (error) return <ErrorAlert error={error} />;

  return (
    <div>
      <h1>{data.user.name}</h1>
      <button onClick={() => refetch()}>刷新</button>
    </div>
  );
}
```

## 二、useMutation

```typescript
const UPDATE_USER = gql`
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      id
      name
      email
    }
  }
`;

function EditUserForm({ user }) {
  const [updateUser, { loading, error }] = useMutation(UPDATE_USER, {
    // 完成后重新查询
    refetchQueries: ['GetUser'],
    // 错误时回滚乐观更新
    onError: (error) => console.error(error),
  });

  const handleSubmit = async (values) => {
    try {
      await updateUser({
        variables: {
          id: user.id,
          input: values,
        },
      });
      // 成功处理
    } catch (e) {
      // 错误处理
    }
  };

  return <form onSubmit={handleSubmit}>...</form>;
}
```

## 三、useSubscription

```typescript
const ORDER_SUBSCRIPTION = gql`
  subscription OnOrderCreated {
    orderCreated {
      id
      total
      customer { name }
    }
  }
`;

function OrderNotifications() {
  const { data, loading } = useSubscription(ORDER_SUBSCRIPTION, {
    onData: ({ data }) => {
      // 新订单通知
      toast.success(`新订单: ${data.data.orderCreated.id}`);
    },
    onError: (error) => {
      console.error('订阅错误:', error);
    },
  });

  if (loading) return <span>等待中...</span>;

  return <div>监听中...</div>;
}
```

## 四、useLazyQuery

```typescript
const SEARCH_USERS = gql`
  query SearchUsers($keyword: String!) {
    users(search: $keyword) {
      edges { node { id name } }
    }
  }
`;

function SearchUser() {
  const [searchUsers, { data, loading }] = useLazyQuery(SEARCH_USERS);

  const handleSearch = (keyword) => {
    searchUsers({ variables: { keyword } });
  };

  return (
    <div>
      <input onChange={(e) => handleSearch(e.target.value)} />
      {loading && <Spinner />}
      {data && data.users.edges.map(/* ... */)}
    </div>
  );
}
```

## 五、注意事项

1. **skip 可以条件执行查询**
2. **pollInterval 做轮询**
3. **fetchMore 做无限滚动**
4. **useLazyQuery 适合搜索场景**
5. **订阅要处理断线重连**
