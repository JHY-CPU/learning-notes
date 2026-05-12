# 模块化Store组织

## 一、概念说明

大型项目中按功能模块拆分Store，每个模块独立一个文件。Pinia天然支持模块化，无需命名空间。

```
src/stores/
  index.js            # 导出所有store
  user.js             # 用户相关
  cart.js             # 购物车
  products.js         # 商品
  notifications.js    # 通知
```

```js
// stores/index.js
export { useUserStore } from './user'
export { useCartStore } from './cart'
export { useProductStore } from './products'
```

## 二、具体用法

### Store文件结构

```js
// stores/user.js
export const useUserStore = defineStore('user', () => {
  const profile = ref(null)
  const isLoggedIn = computed(() => !!profile.value)
  const login = async (creds) => { /* ... */ }
  return { profile, isLoggedIn, login }
})

// stores/cart.js
import { useUserStore } from './user'

export const useCartStore = defineStore('cart', () => {
  const items = ref([])

  const checkout = async () => {
    const user = useUserStore()
    if (!user.isLoggedIn) throw new Error('请先登录')
    // ...
  }

  return { items, checkout }
})
```

### 按功能子目录组织

```
src/stores/
  auth/
    index.js
    user.js
    permissions.js
  shop/
    index.js
    products.js
    cart.js
    orders.js
```

## 四、大型项目Store目录结构

```
src/stores/
  index.ts                  # 创建并导出 pinia 实例
  types.ts                  # 共享类型定义
  
  auth/
    index.ts                # 导出 useAuth
    user.ts                 # 用户信息 store
    permissions.ts          # 权限 store
    actions.ts              # 认证相关 action
  
  shop/
    index.ts                # 导出所有 shop store
    products.ts             # 商品 store
    cart.ts                 # 购物车 store
    orders.ts               # 订单 store
  
  app/
    index.ts
    theme.ts                # 主题 store
    locale.ts               # 语言 store
    sidebar.ts              # 侧边栏状态 store
  
  shared/
    loading.ts              # 全局加载状态
    notifications.ts        # 通知 store
```

## 五、Store组合模式

```ts
// stores/shop/index.ts
import { useProductStore } from './products'
import { useCartStore } from './cart'
import { useOrderStore } from './orders'

// 组合Store：提供统一接口
export function useShop() {
  const products = useProductStore()
  const cart = useCartStore()
  const orders = useOrderStore()

  // 业务流程封装
  async function purchase(productId: number) {
    const product = products.getById(productId)
    if (!product) throw new Error('商品不存在')

    cart.addItem(product)
    const order = await orders.createOrder(cart.items)
    cart.clearItems()

    return order
  }

  return {
    products,
    cart,
    orders,
    purchase
  }
}
```

## 六、Store懒加载

```ts
// stores/lazy.ts
import { defineStore } from 'pinia'

// 大型Store可以按需加载
export const useHeavyStore = defineStore('heavy', () => {
  const data = ref([])

  async function loadData() {
    // 只在需要时才加载数据
    if (data.value.length === 0) {
      data.value = await api.fetchHeavyData()
    }
  }

  return { data, loadData }
})

// 或者使用动态 import
const useAnalyticsStore = () => import('./analytics').then(m => m.useAnalyticsStore())
```

## 三、注意事项与常见陷阱

1. 每个Store文件只负责一个功能领域
2. Store ID与文件名保持一致
3. 使用统一的导出入口（index.js）
4. Store间引用使用相对路径或别名
5. 避免Store间过度耦合，保持职责单一
6. 组合Store模式可以封装复杂业务流程
7. 懒加载Store适合不常用的大型模块
