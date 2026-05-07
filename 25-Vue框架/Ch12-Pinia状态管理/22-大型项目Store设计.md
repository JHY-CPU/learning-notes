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

## 三、注意事项与常见陷阱

1. 按业务域而非数据表划分Store
2. 公共逻辑（CRUD、分页）提取为工厂函数
3. 保持Store间的依赖关系清晰（单向依赖）
4. 使用TypeScript增强类型安全
5. 定期重构，避免Store膨胀
