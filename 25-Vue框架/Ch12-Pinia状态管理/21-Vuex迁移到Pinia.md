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

## 三、注意事项与常见陷阱

1. Pinia没有mutations，直接在actions中修改state
2. 不需要`namespaced`，每个Store天然隔离
3. 跨Store调用直接`useXxxStore()`，不需要`rootState`
4. 可以渐进迁移：Vuex和Pinia共存（不推荐长期）
5. 测试代码也需要相应调整
