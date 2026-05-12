# 大型项目Store设计

## 一、概念说明

大型项目中需要合理的Store架构设计：按业务域划分、统一命名规范、公共逻辑提取、Store间依赖管理。

```
src/stores/
  index.ts                  # 统一导出
  types.ts                  # 共享类型
  plugins/                  # 插件
    persist.ts
    logger.ts
  modules/
    auth/
      index.ts              # auth模块导出
      user.ts
      permissions.ts
    shop/
      index.ts
      products.ts
      cart.ts
      orders.ts
    common/
      index.ts
      app.ts
      notifications.ts
```

## 二、具体用法

### 公共逻辑提取

```js
// stores/common/crudFactory.js
export function createCrudStore(name, api) {
  return defineStore(name, () => {
    const items = ref([])
    const loading = ref(false)
    const error = ref(null)

    const fetchAll = async () => {
      loading.value = true
      items.value = await api.getAll()
      loading.value = false
    }

    const create = async (data) => {
      const item = await api.create(data)
      items.value.push(item)
    }

    return { items, loading, error, fetchAll, create }
  })
}

// 使用
export const useProducts = createCrudStore('products', productApi)
export const useOrders = createCrudStore('orders', orderApi)
```

### 命名规范

```
Store命名：use + 业务名 + Store
  useUserStore, useCartStore, useProductStore

文件命名：kebab-case
  user.js, cart.js, product-store.js

ID命名：与文件名一致
  'user', 'cart', 'product'
```

## 四、CRUD Store 工厂

```ts
// stores/factory/createCrudStore.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

interface CrudApi<T> {
  getAll: () => Promise<T[]>
  getById: (id: number) => Promise<T>
  create: (data: Partial<T>) => Promise<T>
  update: (id: number, data: Partial<T>) => Promise<T>
  delete: (id: number) => Promise<void>
}

export function createCrudStore<T extends { id: number }>(
  name: string,
  api: CrudApi<T>
) {
  return defineStore(name, () => {
    const items = ref<T[]>([]) as any
    const current = ref<T | null>(null)
    const loading = ref(false)
    const error = ref<string | null>(null)

    const getById = computed(() => (id: number) =>
      items.value.find((item: T) => item.id === id)
    )

    async function fetchAll() {
      loading.value = true
      error.value = null
      try {
        items.value = await api.getAll()
      } catch (e: any) {
        error.value = e.message
      } finally {
        loading.value = false
      }
    }

    async function create(data: Partial<T>) {
      const item = await api.create(data)
      items.value.push(item)
      return item
    }

    async function update(id: number, data: Partial<T>) {
      const updated = await api.update(id, data)
      const index = items.value.findIndex((i: T) => i.id === id)
      if (index > -1) items.value[index] = updated
      return updated
    }

    async function remove(id: number) {
      await api.delete(id)
      items.value = items.value.filter((i: T) => i.id !== id)
    }

    return {
      items, current, loading, error,
      getById, fetchAll, create, update, remove
    }
  })
}

// 使用
// stores/products.ts
import { createCrudStore } from './factory/createCrudStore'
import * as productApi from '@/api/products'
export const useProductStore = createCrudStore('products', productApi)

// stores/orders.ts
import { createCrudStore } from './factory/createCrudStore'
import * as orderApi from '@/api/orders'
export const useOrderStore = createCrudStore('orders', orderApi)
```

## 五、Store间依赖图

```
依赖方向（单向）：

auth ←─────┐
  ↓         │
cart ───────┤
  ↓         │
order ──────┘
  ↓
notification

规则：
- auth 不依赖任何 Store
- cart 依赖 auth
- order 依赖 auth, cart
- notification 被多个 Store 调用，但不依赖其他 Store
```

## 三、注意事项与常见陷阱

1. 按业务域而非数据表划分Store
2. 公共逻辑（CRUD、分页）提取为工厂函数
3. 保持Store间的依赖关系清晰（单向依赖）
4. 使用TypeScript增强类型安全
5. 定期重构，避免Store膨胀
6. 工厂函数可以大幅减少重复的 CRUD 代码
7. 依赖关系图有助于发现循环依赖和设计问题
