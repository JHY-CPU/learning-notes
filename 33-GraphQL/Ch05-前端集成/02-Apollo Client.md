# Apollo Client

## 一、核心概念

```typescript
import { ApolloClient, InMemoryCache, gql, useQuery, useMutation } from '@apollo/client';

// 定义查询
const GET_USERS = gql`
  query GetUsers($limit: Int!) {
    users(first: $limit) {
      edges {
        node {
          id
          name
          email
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
`;

// 在组件中使用
function UserList() {
  const { data, loading, error } = useQuery(GET_USERS, {
    variables: { limit: 10 },
    fetchPolicy: 'cache-first',
  });

  if (loading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return data.users.edges.map(({ node }) => (
    <UserCard key={node.id} user={node} />
  ));
}
```

## 二、Mutation 使用

```typescript
const CREATE_USER = gql`
  mutation CreateUser($input: CreateUserInput!) {
    createUser(input: $input) {
      id
      name
      email
    }
  }
`;

function CreateUserForm() {
  const [createUser, { loading }] = useMutation(CREATE_USER, {
    // 乐观更新
    optimisticResponse: ({ input }) => ({
      createUser: {
        __typename: 'User',
        id: 'temp-id',
        ...input,
      },
    }),
    // 更新缓存
    update(cache, { data: { createUser } }) {
      cache.modify({
        fields: {
          users(existing = []) {
            const newUserRef = cache.writeFragment({
              data: createUser,
              fragment: gql`
                fragment NewUser on User {
                  id name email
                }
              `,
            });
            return [...existing, newUserRef];
          },
        },
      });
    },
  });

  const handleSubmit = async (input) => {
    await createUser({ variables: { input } });
  };

  return <form onSubmit={handleSubmit}>...</form>;
}
```

## 三、缓存策略

```typescript
// fetchPolicy 选项
useQuery(GET_USERS, {
  fetchPolicy: 'cache-first',      // 默认：先缓存后网络
  // 'cache-only'                   // 只用缓存
  // 'cache-and-network'            // 缓存 + 网络并行
  // 'network-only'                 // 只用网络
  // 'no-cache'                     // 不缓存
  // 'standby'                      // 不自动执行
});

// nextFetchPolicy
useQuery(GET_USERS, {
  fetchPolicy: 'cache-and-network',
  nextFetchPolicy: 'cache-first',  // 后续用缓存优先
});
```

## 四、注意事项

1. **gql 模板标签定义查询**
2. **变量通过 options 传递**
3. **fetchPolicy 要根据场景选择**
4. **乐观更新提升用户体验**
5. **cache.modify 精细更新缓存**
