# Vue TS最佳实践

## 一、概念说明

Vue 3 + TypeScript 的最佳实践围绕类型推断最大化、代码简洁性和团队协作效率。核心原则是"让 TypeScript 做它擅长的事"——自动推断、编译时检查、IDE 支持。

## 二、具体用法

### 2.1 Props 设计原则

```vue
<!-- 好的 Props 设计 -->
<script setup lang="ts">
interface Props {
  // 1. 使用联合类型限制取值
  variant: 'primary' | 'secondary' | 'danger';

  // 2. 可选属性用 ? 标记
  size?: 'sm' | 'md' | 'lg';

  // 3. 复杂对象用 interface 定义
  user?: User;

  // 4. 回调函数有明确签名
  onSelect?: (item: Item) => void;
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
});
</script>
```

### 2.2 组合式函数最佳实践

```typescript
// composables/useApi.ts — 泛型组合式函数
export function useApi<T>(endpoint: string) {
  const data = shallowRef<T | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);

  const fetch = async () => {
    loading.value = true;
    try {
      data.value = await api.get<T>(endpoint);
    } catch (e) {
      error.value = e instanceof Error ? e.message : '未知错误';
    } finally {
      loading.value = false;
    }
  };

  // 返回 as const 确保精确类型
  return { data, loading, error, fetch } as const;
}
```

### 2.3 类型导出策略

```typescript
// types/index.ts
export type { User, CreateUserDto } from './user';
export type { ApiResponse, PaginatedResponse } from './api';

// 使用 import type
import type { User } from '../types';
```

### 2.4 Store 设计

```typescript
// stores/user.ts — Setup Store 推荐
export const useUserStore = defineStore('user', () => {
  const user = ref<User | null>(null);
  const isLoggedIn = computed(() => user.value !== null);

  const login = async (credentials: LoginCredentials) => {
    user.value = await api.login(credentials);
  };

  const logout = () => {
    user.value = null;
  };

  return { user, isLoggedIn, login, logout };
});
```

### 2.5 组件组织

```
components/
├── UserCard/
│   ├── UserCard.vue          # 组件
│   ├── UserCard.test.ts      # 测试
│   ├── UserCard.stories.ts   # Storybook
│   └── index.ts              # 导出
├── DataTable/
│   ├── DataTable.vue
│   ├── types.ts              # 类型定义
│   └── index.ts
└── index.ts                  # 统一导出
```

### 2.6 常见模式

```vue
<script setup lang="ts">
// 模式一：解构 props
const { user, variant } = toRefs(props);

// 模式二：计算派生数据
const displayName = computed(() =>
  user.value ? `${user.value.firstName} ${user.value.lastName}` : '未知用户'
);

// 模式三：响应式表单
const form = reactive({
  username: '',
  password: '',
  remember: false,
});

// 模式四：副作用清理
watch(userId, () => {
  const controller = new AbortController();
  // ...

  onUnmounted(() => controller.abort());
});
</script>
```

## 三、注意事项与常见陷阱

1. **优先使用 Composition API**：类型推断比 Options API 好得多
2. **不要过度类型化**：能推断的类型不需要手动标注
3. **使用 `import type`**：确保 `isolatedModules` 兼容
4. **组件 Props 用 `interface`**：可被扩展
5. **组合式函数返回 `as const`**：确保精确类型
6. **使用 `vue-tsc` 做类型检查**：而不是 `tsc`
7. **保持依赖版本同步**：Vue、vue-tsc、@vue/test-utils
