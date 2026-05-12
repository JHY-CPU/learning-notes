# Store插件

## 一、概念说明

Pinia插件可以扩展Store的功能：添加全局属性、状态持久化、日志记录等。插件是一个函数，接收`context`参数。

```js
// plugins/logger.js
export function loggerPlugin({ store }) {
  store.$onAction(({ name, args, after, onError }) => {
    console.log(`[${store.$id}] Action: ${name}`, args)
    after((result) => console.log(`[${store.$id}] 完成:`, result))
    onError((error) => console.error(`[${store.$id}] 失败:`, error))
  })
}

// main.js
pinia.use(loggerPlugin)
```

## 二、具体用法

### 插件context属性

```js
pinia.use(({ store, app, pinia, options }) => {
  store.$id       // Store ID
  store.$state    // 当前状态
  store.$patch    // 批量修改
  store.$subscribe // 订阅状态变化
  store.$onAction  // 订阅action调用

  app             // Vue应用实例
  pinia           // Pinia实例
  options         // defineStore的配置
})
```

### 添加全局属性

```js
pinia.use(({ store }) => {
  // 添加公共方法
  store.$resetAll = () => {
    store.$patch(store.$state)
  }

  // 添加环境信息
  store.$env = import.meta.env.MODE
})
```

### 条件插件

```js
pinia.use(({ store }) => {
  // 只对特定store生效
  if (store.$id === 'auth') {
    store.$subscribe((mutation, state) => {
      localStorage.setItem('auth', JSON.stringify(state))
    })
  }
})
```

## 四、状态重置插件

```js
// plugins/resetPlugin.js
export function resetPlugin({ store }) {
  // 保存初始状态
  const initialState = JSON.parse(JSON.stringify(store.$state))

  // 添加 $resetAll 方法
  store.$resetAll = () => {
    store.$patch(initialState)
  }
}

// 使用
const store = useUserStore()
store.$resetAll()
```

## 五、开发调试插件

```js
// plugins/devtools.js
export function devtoolsPlugin({ store, app }) {
  if (!import.meta.env.DEV) return

  // 记录所有状态变化
  store.$subscribe((mutation, state) => {
    console.group(`[Pinia] ${mutation.storeId} - ${mutation.type}`)
    console.log('State:', JSON.parse(JSON.stringify(state)))
    console.groupEnd()
  })

  // 记录所有 action 调用
  store.$onAction(({ name, args, after, onError }) => {
    const start = Date.now()
    console.log(`[Action] ${store.$id}.${name}`, args)

    after((result) => {
      console.log(`[Action] ${store.$id}.${name} 完成 (${Date.now() - start}ms)`)
    })

    onError((error) => {
      console.error(`[Action] ${store.$id}.${name} 失败:`, error)
    })
  })
}
```

## 六、插件工厂函数

```js
// plugins/persistPlugin.js
export function createPersistPlugin(options = {}) {
  const { key = 'pinia', paths = [] } = options

  return ({ store }) => {
    // 从 localStorage 恢复
    const saved = localStorage.getItem(`${key}:${store.$id}`)
    if (saved) {
      store.$patch(JSON.parse(saved))
    }

    // 订阅变化并保存
    store.$subscribe((mutation, state) => {
      const toSave = paths.length
        ? Object.fromEntries(paths.map(p => [p, state[p]]))
        : state
      localStorage.setItem(`${key}:${store.$id}`, JSON.stringify(toSave))
    })
  }
}

// 使用
pinia.use(createPersistPlugin({
  key: 'myapp',
  paths: ['token', 'theme']  // 只持久化指定字段
}))
```

## 三、注意事项与常见陷阱

1. 插件在`app.use(pinia)`之前注册
2. 插件中的`$subscribe`和`$onAction`在组件卸载后仍有效
3. 插件添加的属性需在TypeScript中声明扩展类型
4. 避免在插件中产生副作用导致无限循环
5. 复杂插件考虑封装为工厂函数接受配置
6. `$subscribe`的`detached: true`确保组件卸载后仍然监听
7. 持久化插件只保存指定字段，避免存储不必要的数据
