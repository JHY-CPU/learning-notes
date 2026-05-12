# Vuex迁移到Pinia

## 一、概念说明

从Vuex迁移到Pinia需要：重写Store定义、删除mutations、调整模块结构。Pinia设计上兼容Vuex概念，迁移成本较低。

```js
// Vuex (旧)
export default {
  namespaced: true,
  state: () => ({ count: 0 }),
  mutations: {
    SET_COUNT(state, val) { state.count = val }
  },
  actions: {
    increment({ commit }) { commit('SET_COUNT', this.count + 1) }
  }
}

// Pinia (新)
export const useCounter = defineStore('counter', {
  state: () => ({ count: 0 }),
  actions: {
    increment() { this.count++ }  // 无mutations
  }
})
```

## 二、具体用法

### 主要差异

| Vuex | Pinia |
|------|-------|
| `mutations` | 删除（直接在actions中修改） |
| `commit('mutation')` | 直接`this.xxx = yyy` |
| `dispatch('action')` | `this.xxx()` |
| `namespaced: true` | 不需要（自动隔离） |
| `modules` | 多个独立Store |
| `getters` | `getters`（相同） |
| `rootState` | `useOtherStore()` |

### 迁移步骤

```
1. 安装Pinia，替换Vuex
2. 将每个Vuex module改为独立的Pinia Store
3. 删除mutations，action中直接修改state
4. 删除namespaced和modules
5. 替换commit/dispatch为直接调用
6. 更新组件中的store使用方式
```

## 四、代码迁移对照

```js
// ========== Vuex ==========
// store/index.js
import Vue from 'vue'
import Vuex from 'vuex'
import user from './modules/user'
import cart from './modules/cart'

Vue.use(Vuex)
export default new Vuex.Store({
  modules: { user, cart }
})

// store/modules/user.js
export default {
  namespaced: true,
  state: () => ({ token: null, profile: null }),
  getters: {
    isLoggedIn: (state) => !!state.token
  },
  mutations: {
    SET_TOKEN(state, token) { state.token = token },
    SET_PROFILE(state, profile) { state.profile = profile }
  },
  actions: {
    async login({ commit }, creds) {
      const { token, user } = await api.login(creds)
      commit('SET_TOKEN', token)
      commit('SET_PROFILE', user)
    }
  }
}

// 组件中
this.$store.dispatch('user/login', creds)
this.$store.state.user.token
this.$store.getters['user/isLoggedIn']

// ========== Pinia ==========
// stores/user.js
import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({ token: null, profile: null }),
  getters: {
    isLoggedIn: (state) => !!state.token
  },
  actions: {
    async login(creds) {
      const { token, user } = await api.login(creds)
      this.token = token
      this.profile = user
    }
  }
})

// 组件中
const userStore = useUserStore()
await userStore.login(creds)
userStore.token
userStore.isLoggedIn
```

## 五、渐进迁移策略

```
阶段1：安装 Pinia，两个状态管理共存
  npm install pinia
  app.use(pinia)
  app.use(store)  // 保留 Vuex

阶段2：逐模块迁移
  - 先迁移简单的 Store（无命名空间嵌套）
  - 用 Pinia 替换对应的 Vuex module
  - 更新组件中的引用

阶段3：移除 Vuex
  - 所有模块迁移完成后删除 Vuex
  - 删除 store/index.js 和 modules 目录
  - 更新 main.js
```

## 三、注意事项与常见陷阱

1. Pinia没有mutations，直接在actions中修改state
2. 不需要`namespaced`，每个Store天然隔离
3. 跨Store调用直接`useXxxStore()`，不需要`rootState`
4. 可以渐进迁移：Vuex和Pinia共存（不推荐长期）
5. 测试代码也需要相应调整
6. `this.$store` 改为 `useXxxStore()` 的函数调用
7. Vuex 的 `mapState`/`mapGetters` 可用 `storeToRefs` 替代
