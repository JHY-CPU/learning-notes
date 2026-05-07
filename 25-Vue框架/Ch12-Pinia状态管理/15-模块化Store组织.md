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

## 三、注意事项与常见陷阱

1. 每个Store文件只负责一个功能领域
2. Store ID与文件名保持一致
3. 使用统一的导出入口（index.js）
4. Store间引用使用相对路径或别名
5. 避免Store间过度耦合，保持职责单一
