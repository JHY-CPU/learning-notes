# Getters计算属性

## 一、概念说明

Getters是Store的计算属性，基于state派生新值。它们有缓存，只在依赖变化时重新计算。

```js
export const useCartStore = defineStore('cart', {
  state: () => ({
    items: [
      { id: 1, name: '商品A', price: 100, quantity: 2 },
      { id: 2, name: '商品B', price: 200, quantity: 1 }
    ]
  }),
  getters: {
    totalPrice: (state) => {
      return state.items.reduce((sum, item) => sum + item.price * item.quantity, 0)
    },
    itemCount: (state) => state.items.reduce((n, item) => n + item.quantity, 0),
    expensiveItems: (state) => state.items.filter(i => i.price > 150)
  }
})
```

## 二、具体用法

### 访问其他getter

```js
getters: {
  totalPrice: (state) => state.items.reduce((s, i) => s + i.price * i.quantity, 0),
  formattedPrice() {
    return `¥${this.totalPrice.toFixed(2)}`  // 用this访问其他getter
  }
}
```

### 带参数的getter（返回函数）

```js
getters: {
  getItemById: (state) => (id) => {
    return state.items.find(item => item.id === id)
  },
  itemsByCategory: (state) => (category) => {
    return state.items.filter(i => i.category === category)
  }
}

// 使用
const item = cart.getItemById(1)
```

### Setup风格的getter

```js
export const useCart = defineStore('cart', () => {
  const items = ref([])
  const totalPrice = computed(() =>
    items.value.reduce((s, i) => s + i.price * i.quantity, 0)
  )
  return { items, totalPrice }
})
```

## 三、注意事项与常见陷阱

1. getter中用`state`参数访问状态（选项式），不能用`this`
2. 访问其他getter时需用`this`（此时不能用箭头函数）
3. getter是只读的，修改会报警告
4. 带参数的getter每次调用都会执行（无缓存）
5. Setup式Store中使用`computed`替代getter

## 四、Getters 的实际场景

### 4.1 权限判断
```js
export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null,
    permissions: []
  }),
  getters: {
    isAdmin: (state) => state.user?.role === 'admin',
    canEdit: (state) => state.permissions.includes('edit'),
    canDelete: (state) => state.permissions.includes('delete'),
    hasPermission: (state) => (perm) => state.permissions.includes(perm)
  }
})
```

### 4.2 数据过滤与排序
```js
export const useProductStore = defineStore('products', {
  state: () => ({
    products: [],
    sortBy: 'price',
    sortOrder: 'asc',
    filterCategory: ''
  }),
  getters: {
    filteredProducts(state) {
      let result = state.products
      if (state.filterCategory) {
        result = result.filter(p => p.category === state.filterCategory)
      }
      return result
    },
    sortedProducts() {
      return [...this.filteredProducts].sort((a, b) => {
        const modifier = this.sortOrder === 'asc' ? 1 : -1
        return a[this.sortBy] > b[this.sortBy] ? modifier : -modifier
      })
    },
    totalProducts: (state) => state.products.length
  }
})
```

## 五、Getter vs computed 对比

| 特性 | Store Getter | 组件 computed |
|------|-------------|--------------|
| 缓存 | 有 | 有 |
| 作用域 | 全局 | 组件 |
| 其他 getter | `this` 访问 | 直接引用 |
| 参数化 | 返回函数 | 返回函数 |
