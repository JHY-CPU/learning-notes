# uni-app 性能优化

## 一、概念说明

uni-app 性能优化涉及包体积优化、渲染性能、网络请求和内存管理等方面。

```javascript
// 性能优化清单
/*
1. 分包加载 - 减小主包体积
2. 图片优化 - 压缩、懒加载、WebP
3. 列表优化 - 虚拟列表、分页加载
4. 代码优化 - 避免不必要的响应式数据
5. 缓存策略 - 合理使用本地缓存
*/
```

## 二、分包优化

```json
// pages.json 分包配置
{
  "pages": [
    { "path": "pages/index/index" },
    { "path": "pages/login/login" }
  ],
  "subPackages": [
    {
      "root": "pkg-shop",
      "pages": [
        { "path": "list/list" },
        { "path": "detail/detail" }
      ]
    }
  ],
  "preloadRule": {
    "pages/index/index": {
      "network": "all",
      "packages": ["pkg-shop"]
    }
  }
}
```

## 三、列表性能优化

```vue
<template>
  <scroll-view
    scroll-y
    :refresher-enabled="true"
    :refresher-triggered="refreshing"
    @refresherrefresh="onRefresh"
    @scrolltolower="loadMore"
  >
    <!-- 使用 v-if 控制渲染 -->
    <view v-for="item in visibleList" :key="item.id" class="item">
      <!-- 懒加载图片 -->
      <image
        :src="item.cover"
        mode="aspectFill"
        lazy-load
        :placeholder="placeholderImg"
      />
      <text>{{ item.name }}</text>
    </view>
    <view v-if="loading" class="loading">加载中...</view>
  </scroll-view>
</template>

<script>
export default {
  data() {
    return {
      list: [],
      page: 1,
      loading: false,
      refreshing: false,
      placeholderImg: '/static/placeholder.png'
    }
  },
  computed: {
    visibleList() {
      // 只渲染可见区域的数据
      return this.list.slice(0, this.page * 20)
    }
  },
  methods: {
    async loadData() {
      if (this.loading) return
      this.loading = true
      const res = await api.getList({ page: this.page })
      this.list = [...this.list, ...res.data]
      this.loading = false
    },
    async onRefresh() {
      this.refreshing = true
      this.page = 1
      this.list = []
      await this.loadData()
      this.refreshing = false
    },
    async loadMore() {
      this.page++
      await this.loadData()
    }
  }
}
</script>
```

## 四、图片优化

```vue
<template>
  <view>
    <!-- 使用 WebP 格式 -->
    <image src="/static/image.webp" />

    <!-- 图片懒加载 -->
    <image :src="item.image" lazy-load mode="aspectFill" />

    <!-- 控制图片尺寸 -->
    <image
      :src="item.thumbnail"
      :style="{ width: '200rpx', height: '200rpx' }"
    />
  </view>
</template>
```

## 五、代码优化

```vue
<script>
export default {
  data() {
    return {
      // 只将需要响应式的数据放在 data 中
      // 大量的静态数据不要放在 data
      list: []
    }
  },
  // 非响应式数据放在外部
  created() {
    this.cacheMap = new Map()
    this.timer = null
  },
  beforeDestroy() {
    // 清理定时器
    if (this.timer) clearInterval(this.timer)
  },
  methods: {
    // 避免在模板中使用复杂计算
    formatPrice(price) {
      // 使用方法缓存结果
      const key = `price_${price}`
      if (this.cacheMap.has(key)) return this.cacheMap.get(key)
      const result = `¥${price.toFixed(2)}`
      this.cacheMap.set(key, result)
      return result
    }
  }
}
</script>
```

## 六、缓存策略

```javascript
// 请求缓存
const cache = {
  data: {},
  get(key) {
    const item = this.data[key]
    if (!item) return null
    if (Date.now() > item.expire) {
      delete this.data[key]
      return null
    }
    return item.value
  },
  set(key, value, ttl = 5 * 60 * 1000) {
    this.data[key] = { value, expire: Date.now() + ttl }
  }
}

const cachedRequest = async (url, options = {}) => {
  const cacheKey = `${url}_${JSON.stringify(options)}`
  const cached = cache.get(cacheKey)
  if (cached) return cached

  const res = await uni.request({ url, ...options })
  cache.set(cacheKey, res.data)
  return res.data
}
```

## 七、注意事项与常见陷阱

1. **主包限制**：微信小程序主包 2MB，尽量将非首页放入分包
2. **图片资源**：使用 CDN 加载大图片，本地只放小图标
3. **减少 setData**：每次 setData 会触发渲染，合并更新减少调用次数
4. **避免深层嵌套**：模板嵌套过深会影响渲染性能
5. **使用分段加载**：长列表使用分页或虚拟滚动
