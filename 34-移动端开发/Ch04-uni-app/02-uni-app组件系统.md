# uni-app 组件系统

## 一、概念说明

uni-app 内置了丰富的基础组件，同时支持创建自定义组件和使用 uni-ui 等组件库。

```vue
<!-- 基础组件使用 -->
<template>
  <view class="page">
    <!-- 视图容器 -->
    <view class="section">
      <text class="title">基础组件</text>
    </view>

    <!-- 内容组件 -->
    <view class="card">
      <image :src="imageUrl" mode="aspectFill" class="image" />
      <view class="info">
        <text class="name">{{ name }}</text>
        <text class="desc">{{ description }}</text>
      </view>
    </view>
  </view>
</template>
```

## 二、基础组件

### 2.1 视图容器

```vue
<template>
  <view>
    <!-- view - 基础容器 -->
    <view class="container">
      <text>内容</text>
    </view>

    <!-- scroll-view - 滚动容器 -->
    <scroll-view
      scroll-y
      :refresher-enabled="true"
      :refresher-triggered="refreshing"
      @refresherrefresh="onRefresh"
      @scrolltolower="loadMore"
      style="height: 400rpx;"
    >
      <view v-for="item in list" :key="item.id" class="item">
        {{ item.name }}
      </view>
    </scroll-view>

    <!-- swiper - 轮播 -->
    <swiper
      class="swiper"
      :indicator-dots="true"
      :autoplay="true"
      :interval="3000"
    >
      <swiper-item v-for="banner in banners" :key="banner.id">
        <image :src="banner.image" mode="aspectFill" />
      </swiper-item>
    </swiper>
  </view>
</template>
```

### 2.2 表单组件

```vue
<template>
  <view class="form">
    <!-- 输入框 -->
    <view class="input-group">
      <text class="label">用户名</text>
      <input
        v-model="form.username"
        placeholder="请输入用户名"
        maxlength="20"
      />
    </view>

    <!-- 多行输入 -->
    <view class="input-group">
      <textarea
        v-model="form.content"
        placeholder="请输入内容"
        :maxlength="500"
      />
    </view>

    <!-- 选择器 -->
    <picker :range="cities" @change="onCityChange">
      <view class="picker">
        <text>{{ form.city || '请选择城市' }}</text>
      </view>
    </picker>

    <!-- 日期选择 -->
    <picker mode="date" :value="form.date" @change="onDateChange">
      <view class="picker">
        <text>{{ form.date || '请选择日期' }}</text>
      </view>
    </picker>

    <!-- 开关 -->
    <view class="switch-group">
      <text>接收通知</text>
      <switch :checked="form.notify" @change="onNotifyChange" />
    </view>

    <!-- 复选框 -->
    <checkbox-group @change="onHobbiesChange">
      <label v-for="hobby in hobbies" :key="hobby">
        <checkbox :value="hobby" :checked="form.hobbies.includes(hobby)" />
        {{ hobby }}
      </label>
    </checkbox-group>
  </view>
</template>

<script>
export default {
  data() {
    return {
      form: {
        username: '',
        content: '',
        city: '',
        date: '',
        notify: true,
        hobbies: []
      },
      cities: ['北京', '上海', '广州', '深圳'],
      hobbies: ['阅读', '运动', '音乐', '游戏']
    }
  },
  methods: {
    onCityChange(e) {
      this.form.city = this.cities[e.detail.value]
    },
    onDateChange(e) {
      this.form.date = e.detail.value
    },
    onNotifyChange(e) {
      this.form.notify = e.detail.value
    },
    onHobbiesChange(e) {
      this.form.hobbies = e.detail.value
    }
  }
}
</script>
```

## 三、自定义组件

```vue
<!-- components/ProductCard.vue -->
<template>
  <view class="product-card" @click="handleClick">
    <image :src="product.image" mode="aspectFill" class="image" />
    <view class="info">
      <text class="name">{{ product.name }}</text>
      <text class="price">¥{{ product.price }}</text>
      <text class="sales">已售 {{ product.sales }}</text>
    </view>
  </view>
</template>

<script>
export default {
  name: 'ProductCard',
  props: {
    product: {
      type: Object,
      required: true
    }
  },
  methods: {
    handleClick() {
      this.$emit('click', this.product)
    }
  }
}
</script>

<style scoped>
.product-card {
  display: flex;
  padding: 20rpx;
  background: #fff;
  border-radius: 12rpx;
  margin-bottom: 16rpx;
}
.image {
  width: 200rpx;
  height: 200rpx;
  border-radius: 8rpx;
}
.info {
  flex: 1;
  margin-left: 20rpx;
}
.name {
  font-size: 28rpx;
  color: #333;
}
.price {
  font-size: 32rpx;
  color: #ff4444;
  font-weight: bold;
}
.sales {
  font-size: 24rpx;
  color: #999;
}
</style>

<!-- 使用组件 -->
<template>
  <view>
    <product-card
      v-for="product in products"
      :key="product.id"
      :product="product"
      @click="goDetail"
    />
  </view>
</template>

<script>
import ProductCard from '@/components/ProductCard.vue'

export default {
  components: { ProductCard },
  methods: {
    goDetail(product) {
      uni.navigateTo({ url: `/pages/product/detail?id=${product.id}` })
    }
  }
}
</script>
```

## 四、注意事项与常见陷阱

1. **组件命名**：组件文件名使用 PascalCase，模板中使用 kebab-case
2. **样式隔离**：使用 scoped 避免样式污染
3. **rpx 使用**：组件内部使用 rpx 确保跨平台一致性
4. **事件绑定**：使用 @click 而非 onclick，使用 @change 监听变化
5. **props 类型检查**：始终定义 props 的类型和默认值
