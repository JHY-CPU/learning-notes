# 定义Store-选项式写法

## 一、概念说明

选项式写法使用`state`、`getters`、`actions`三个选项定义Store，与Vuex的写法类似但更简洁（无mutations）。

```js
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    name: '张三',
    age: 20,
    isLoggedIn: false,
    token: null
  }),
  getters: {
    info: (state) => `${state.name}, ${state.age}岁`,
    isAdmin: (state) => state.token === 'admin-token'
  },
  actions: {
    login(token) {
      this.token = token
      this.isLoggedIn = true
    },
    logout() {
      this.token = null
      this.isLoggedIn = false
    }
  }
})
```

## 二、具体用法

### 使用Store

```vue
<script setup>
import { useUserStore } from '@/stores/user'

const user = useUserStore()

// 读取state
console.log(user.name)
console.log(user.info)  // getter

// 修改state（直接修改或通过action）
user.login('abc123')
user.name = '李四'  // 直接修改也行
</script>
```

### 带参数的getter

```js
getters: {
  // getter不能直接传参，返回函数
  getUserById: (state) => {
    return (id) => state.users.find(u => u.id === id)
  }
}

// 使用
const store = useStore()
store.getUserById(123)  // 返回id为123的用户
```

## 四、完整选项式Store示例

```js
import { defineStore } from 'pinia'

export const useProductStore = defineStore('products', {
  state: () => ({
    items: [],
    categories: [],
    filter: { category: '', priceRange: [0, 1000] },
    sortBy: 'name',
    loading: false
  }),

  getters: {
    filteredItems: (state) => {
      let result = state.items

      if (state.filter.category) {
        result = result.filter(item => item.category === state.filter.category)
      }

      result = result.filter(item =>
        item.price >= state.filter.priceRange[0] &&
        item.price <= state.filter.priceRange[1]
      )

      // 排序
      result.sort((a, b) => {
        if (state.sortBy === 'price') return a.price - b.price
        return a.name.localeCompare(b.name)
      })

      return result
    },

    totalPrice: (state) => state.items.reduce((sum, item) => sum + item.price, 0),

    // 带参数的 getter
    getById: (state) => (id) => state.items.find(item => item.id === id)
  },

  actions: {
    async fetchProducts() {
      this.loading = true
      try {
        const { data } = await api.getProducts()
        this.items = data
      } finally {
        this.loading = false
      }
    },

    // 使用 $patch 批量修改
    setFilter(filter) {
      this.$patch({ filter })
    },

    // $reset 重置到初始状态
    resetFilters() {
      this.$reset()
    }
  }
})
```

## 三、注意事项与常见陷阱

1. `state`必须是函数（返回对象），保证每个Store实例独立
2. `getters`中通过`state`参数访问状态，不能用`this`
3. `actions`中用`this`访问整个store（state + getters + actions）
4. Pinia没有`mutations`，直接在actions中修改状态
5. getter返回的函数需要再调用一次才能获取值
6. `$patch`可以批量修改多个属性，性能更优
7. 选项式写法可用`this.$patch`、`this.$reset`等方法
