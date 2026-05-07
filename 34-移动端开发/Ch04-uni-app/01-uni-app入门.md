# uni-app 入门

## 一、概念说明

uni-app 是 DCloud 推出的跨平台开发框架，使用 Vue.js 语法，一套代码可发布到 iOS、Android、Web 以及各种小程序平台（微信、支付宝、百度、头条等）。

```vue
<!-- 基础页面结构 -->
<template>
  <view class="container">
    <text class="title">{{ title }}</text>
    <button @click="handleClick">点击我</button>
  </view>
</template>

<script>
export default {
  data() {
    return {
      title: 'Hello uni-app'
    }
  },
  methods: {
    handleClick() {
      uni.showToast({ title: '按钮被点击' })
    }
  }
}
</script>

<style>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20rpx;
}
.title {
  font-size: 36rpx;
  color: #333;
}
</style>
```

## 二、项目创建与结构

```bash
# 使用 HBuilderX 创建项目
# 或使用 CLI
npm install -g @dcloudio/uni-cli
uni create my-app

# 项目结构
/*
my-app/
├── pages/              // 页面
│   └── index/index.vue
├── components/         // 组件
├── static/             // 静态资源
├── store/              // Vuex 状态管理
├── utils/              // 工具函数
├── App.vue             // 应用入口
├── main.js             // 主入口
├── manifest.json       // 应用配置
├── pages.json          // 页面路由配置
└── uni.scss            // 全局样式变量
*/
```

## 三、页面配置

```json
// pages.json
{
  "pages": [
    {
      "path": "pages/index/index",
      "style": {
        "navigationBarTitleText": "首页",
        "navigationBarBackgroundColor": "#007AFF",
        "navigationBarTextStyle": "white",
        "enablePullDownRefresh": true
      }
    },
    {
      "path": "pages/detail/detail",
      "style": {
        "navigationBarTitleText": "详情"
      }
    }
  ],
  "tabBar": {
    "color": "#999",
    "selectedColor": "#007AFF",
    "list": [
      {
        "pagePath": "pages/index/index",
        "text": "首页",
        "iconPath": "static/tabbar/home.png",
        "selectedIconPath": "static/tabbar/home-active.png"
      },
      {
        "pagePath": "pages/mine/mine",
        "text": "我的",
        "iconPath": "static/tabbar/mine.png",
        "selectedIconPath": "static/tabbar/mine-active.png"
      }
    ]
  },
  "globalStyle": {
    "navigationBarTextStyle": "black",
    "navigationBarTitleText": "uni-app",
    "navigationBarBackgroundColor": "#F8F8F8",
    "backgroundColor": "#F8F8F8"
  }
}
```

## 四、生命周期

```vue
<template>
  <view>{{ message }}</view>
</template>

<script>
export default {
  data() {
    return { message: '页面生命周期' }
  },
  // 应用生命周期 (App.vue)
  onLaunch() {
    console.log('应用启动')
  },
  onShow() {
    console.log('应用进入前台')
  },
  onHide() {
    console.log('应用进入后台')
  },

  // 页面生命周期
  onLoad(options) {
    console.log('页面加载', options)
  },
  onShow() {
    console.log('页面显示')
  },
  onReady() {
    console.log('页面初次渲染完成')
  },
  onHide() {
    console.log('页面隐藏')
  },
  onUnload() {
    console.log('页面卸载')
  },
  onPullDownRefresh() {
    console.log('下拉刷新')
    setTimeout(() => uni.stopPullDownRefresh(), 1000)
  },
  onReachBottom() {
    console.log('触底加载更多')
  },
  onShareAppMessage() {
    return {
      title: '分享标题',
      path: '/pages/index/index'
    }
  }
}
</script>
```

## 五、uni-app API

```vue
<script>
export default {
  methods: {
    // 导航
    navigateTo() {
      uni.navigateTo({ url: '/pages/detail/detail?id=1' })
    },
    redirectTo() {
      uni.redirectTo({ url: '/pages/login/login' })
    },
    switchTab() {
      uni.switchTab({ url: '/pages/mine/mine' })
    },

    // 交互
    showToast() {
      uni.showToast({ title: '操作成功', icon: 'success' })
    },
    showModal() {
      uni.showModal({
        title: '提示',
        content: '确定要删除吗？',
        success: (res) => {
          if (res.confirm) console.log('确定')
        }
      })
    },

    // 网络请求
    async fetchData() {
      const [err, res] = await uni.request({
        url: 'https://api.example.com/data',
        method: 'GET'
      })
      if (!err) console.log(res.data)
    },

    // 本地存储
    async saveData() {
      await uni.setStorageSync('key', 'value')
      const data = uni.getStorageSync('key')
    },

    // 选择图片
    chooseImage() {
      uni.chooseImage({
        count: 1,
        success: (res) => {
          this.imagePath = res.tempFilePaths[0]
        }
      })
    }
  }
}
</script>
```

## 六、条件编译

```vue
<template>
  <view>
    <!-- #ifdef APP-PLUS -->
    <text>仅 App 显示</text>
    <!-- #endif -->

    <!-- #ifdef MP-WEIXIN -->
    <text>仅微信小程序显示</text>
    <!-- #endif -->

    <!-- #ifdef H5 -->
    <text>仅 H5 显示</text>
    <!-- #endif -->
  </view>
</template>

<script>
export default {
  methods: {
    platformSpecific() {
      // #ifdef APP-PLUS
      plus.runtime.openURL('https://example.com')
      // #endif

      // #ifdef MP-WEIXIN
      wx.navigateTo({ url: '/pages/detail/detail' })
      // #endif
    }
  }
}
</script>
```

## 七、注意事项与常见陷阱

1. **rpx 单位**：使用 rpx 而非 px，自动适配不同屏幕
2. **条件编译**：不同平台 API 不同，使用条件编译处理差异
3. **组件规范**：uni-app 中使用 view/text/image，不要使用 div/span/img
4. **平台限制**：部分功能在某些平台不可用，提前了解各平台限制
5. **性能注意**：避免在页面中使用过多计算属性，影响渲染性能
