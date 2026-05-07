# uni-app 状态管理

## 一、概念说明

uni-app 支持 Vuex 和 Pinia 两种状态管理方案。Vuex 是传统的 Vue 状态管理库，Pinia 是 Vue 3 推荐的新方案。

```javascript
// store/index.js - Vuex 基础
import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const store = new Vuex.Store({
  state: {
    token: uni.getStorageSync('token') || '',
    userInfo: null,
    cart: []
  },
  mutations: {
    SET_TOKEN(state, token) {
      state.token = token
      uni.setStorageSync('token', token)
    },
    SET_USER(state, user) {
      state.userInfo = user
    },
    ADD_TO_CART(state, product) {
      const existing = state.cart.find(item => item.id === product.id)
      if (existing) {
        existing.quantity++
      } else {
        state.cart.push({ ...product, quantity: 1 })
      }
    },
    REMOVE_FROM_CART(state, id) {
      state.cart = state.cart.filter(item => item.id !== id)
    }
  },
  getters: {
    cartCount: state => state.cart.reduce((sum, item) => sum + item.quantity, 0),
    cartTotal: state => state.cart.reduce((sum, item) => sum + item.price * item.quantity, 0),
    isLoggedIn: state => !!state.token
  },
  actions: {
    async login({ commit }, { username, password }) {
      const res = await uni.request({
        url: '/api/login',
        method: 'POST',
        data: { username, password }
      })
      commit('SET_TOKEN', res.data.token)
      commit('SET_USER', res.data.user)
    },
    async logout({ commit }) {
      uni.removeStorageSync('token')
      commit('SET_TOKEN', '')
      commit('SET_USER', null)
    }
  }
})

export default store
```

## 二、Vuex 使用

### 2.1 模块化

```javascript
// store/modules/user.js
const user = {
  namespaced: true,
  state: {
    info: null,
    addresses: []
  },
  mutations: {
    SET_INFO(state, info) { state.info = info },
    SET_ADDRESSES(state, addresses) { state.addresses = addresses }
  },
  actions: {
    async fetchInfo({ commit }) {
      const res = await uni.request({ url: '/api/user/info' })
      commit('SET_INFO', res.data)
    },
    async fetchAddresses({ commit }) {
      const res = await uni.request({ url: '/api/user/addresses' })
      commit('SET_ADDRESSES', res.data)
    }
  }
}
export default user

// store/index.js
import user from './modules/user'
const store = new Vuex.Store({
  modules: { user }
})
```

### 2.2 组件中使用

```vue
<template>
  <view>
    <text>用户: {{ userInfo?.name }}</text>
    <text>购物车: {{ cartCount }} 件</text>
    <button @click="handleLogin">登录</button>
  </view>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  computed: {
    ...mapState(['userInfo', 'cart']),
    ...mapGetters(['cartCount', 'cartTotal', 'isLoggedIn']),
    ...mapState('user', { addresses: state => state.user.addresses })
  },
  methods: {
    ...mapActions(['login', 'logout']),
    ...mapActions('user', ['fetchInfo']),
    async handleLogin() {
      await this.login({ username: 'test', password: '123456' })
      uni.showToast({ title: '登录成功' })
    }
  }
}
</script>
```

## 三、Pinia（推荐）

```javascript
// stores/user.js
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: uni.getStorageSync('token') || '',
    info: null
  }),
  getters: {
    isLoggedIn: (state) => !!state.token,
    displayName: (state) => state.info?.nickname || '未登录'
  },
  actions: {
    async login(username, password) {
      const res = await uni.request({
        url: '/api/login',
        method: 'POST',
        data: { username, password }
      })
      this.token = res.data.token
      this.info = res.data.user
      uni.setStorageSync('token', this.token)
    },
    logout() {
      this.token = ''
      this.info = null
      uni.removeStorageSync('token')
    }
  }
})

// stores/cart.js
export const useCartStore = defineStore('cart', {
  state: () => ({
    items: []
  }),
  getters: {
    count: (state) => state.items.reduce((s, i) => s + i.quantity, 0),
    total: (state) => state.items.reduce((s, i) => s + i.price * i.quantity, 0)
  },
  actions: {
    addItem(product) {
      const existing = this.items.find(i => i.id === product.id)
      if (existing) {
        existing.quantity++
      } else {
        this.items.push({ ...product, quantity: 1 })
      }
    },
    removeItem(id) {
      this.items = this.items.filter(i => i.id !== id)
    }
  }
})
```

```vue
<!-- 使用 Pinia -->
<template>
  <view>
    <text>{{ userStore.displayName }}</text>
    <text>购物车: {{ cartStore.count }} 件</text>
    <button @click="handleLogin">登录</button>
    <button @click="addToCart">加入购物车</button>
  </view>
</template>

<script setup>
import { useUserStore } from '@/stores/user'
import { useCartStore } from '@/stores/cart'

const userStore = useUserStore()
const cartStore = useCartStore()

const handleLogin = async () => {
  await userStore.login('test', '123456')
  uni.showToast({ title: '登录成功' })
}

const addToCart = () => {
  cartStore.addItem({ id: 1, name: '商品', price: 99 })
}
</script>
```

## 四、注意事项与常见陷阱

1. **状态持久化**：重要状态（如 token）需要同步存储到本地
2. **避免直接修改 state**：Vuex 中必须通过 mutation 修改 state
3. **模块命名空间**：大型项目使用 namespaced: true 避免命名冲突
4. **异步操作**：异步逻辑放在 action 中，不要放在 mutation 中
5. **性能优化**：使用 mapGetters/selective binding 只订阅需要的状态
