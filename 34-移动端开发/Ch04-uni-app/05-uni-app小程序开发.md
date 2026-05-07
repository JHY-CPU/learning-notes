# uni-app 小程序开发

## 一、概念说明

uni-app 可以一套代码编译到微信、支付宝、百度、抖音等多个小程序平台。需要了解各平台的差异和条件编译机制。

```javascript
// 平台标识
/*
MP-WEIXIN     微信小程序
MP-ALIPAY     支付宝小程序
MP-BAIDU      百度小程序
MP-TOUTIAO    抖音小程序
MP-QQ         QQ 小程序
APP-PLUS      App
H5            H5 网页
*/
```

## 二、微信小程序特有功能

### 2.1 微信登录

```javascript
// 微信登录流程
const wxLogin = async () => {
  // 1. 获取微信登录 code
  const loginRes = await uni.login({ provider: 'weixin' })

  // 2. 将 code 发送到后端换取 openid/session_key
  const res = await uni.request({
    url: '/api/auth/wxlogin',
    method: 'POST',
    data: { code: loginRes.code }
  })

  // 3. 存储登录态
  uni.setStorageSync('token', res.data.token)
}

// 获取用户信息（新版需使用按钮触发）
const getUserProfile = () => {
  uni.getUserProfile({
    desc: '用于完善用户资料',
    success: (res) => {
      console.log(res.userInfo)
    }
  })
}
```

### 2.2 微信支付

```javascript
const wxPay = async (orderId) => {
  // 从后端获取支付参数
  const payParams = await api.post('/pay/wxpay', { orderId })

  // 发起支付
  const result = await uni.requestPayment({
    provider: 'wxpay',
    orderInfo: {
      appid: payParams.appid,
      partnerid: payParams.partnerid,
      prepayid: payParams.prepayid,
      package: payParams.package,
      noncestr: payParams.noncestr,
      timestamp: payParams.timestamp,
      sign: payParams.sign
    }
  })

  if (result.errMsg === 'requestPayment:ok') {
    uni.showToast({ title: '支付成功' })
  }
}
```

### 2.3 小程序分享

```vue
<script>
export default {
  // 分享给朋友
  onShareAppMessage() {
    return {
      title: '推荐给你一个好东西',
      path: '/pages/index/index',
      imageUrl: '/static/share.png'
    }
  },
  // 分享到朋友圈
  onShareTimeline() {
    return {
      title: '推荐给你一个好东西',
      query: 'from=timeline',
      imageUrl: '/static/share.png'
    }
  }
}
</script>
```

## 三、条件编译处理平台差异

```vue
<template>
  <view>
    <!-- 模板中的条件编译 -->
    <!-- #ifdef MP-WEIXIN -->
    <button open-type="share">分享给朋友</button>
    <!-- #endif -->

    <!-- #ifdef MP-ALIPAY -->
    <button @click="aliShare">分享</button>
    <!-- #endif -->
  </view>
</template>

<script>
export default {
  methods: {
    async login() {
      // #ifdef MP-WEIXIN
      const res = await uni.login({ provider: 'weixin' })
      // #endif

      // #ifdef MP-ALIPAY
      const res = await uni.login({ provider: 'alipay' })
      // #endif

      // #ifdef APP-PLUS
      const res = await uni.login({ provider: 'weixin' })
      // #endif
    },

    getLocation() {
      uni.getLocation({
        // #ifdef MP-WEIXIN
        type: 'gcj02',
        // #endif
        // #ifdef MP-ALIPAY
        type: 1,
        // #endif
        success: (res) => {
          console.log(res.latitude, res.longitude)
        }
      })
    }
  }
}
</script>
```

## 四、小程序发布流程

```bash
# 1. 在 manifest.json 配置小程序 appid

# 2. HBuilderX 中发行
# 菜单 -> 发行 -> 小程序-微信

# 3. 或使用 CLI
npm run build:mp-weixin

# 4. 在微信开发者工具中打开 dist/dev/mp-weixin 目录
# 5. 上传代码并提交审核
```

```json
// manifest.json 微信小程序配置
{
  "mp-weixin": {
    "appid": "wx1234567890abcdef",
    "setting": {
      "urlCheck": true,
      "es6": true,
      "postcss": true,
      "minified": true
    },
    "usingComponents": true,
    "optimization": {
      "subPackages": true
    }
  }
}
```

## 五、分包加载

```json
// pages.json
{
  "pages": [
    { "path": "pages/index/index" },
    { "path": "pages/mine/mine" }
  ],
  "subPackages": [
    {
      "root": "pages-shop",
      "pages": [
        { "path": "list/list" },
        { "path": "detail/detail" }
      ]
    },
    {
      "root": "pages-order",
      "pages": [
        { "path": "list/list" },
        { "path": "detail/detail" }
      ]
    }
  ],
  "preloadRule": {
    "pages/index/index": {
      "network": "all",
      "packages": ["pages-shop"]
    }
  }
}
```

## 六、注意事项与常见陷阱

1. **包体积限制**：微信小程序主包不超过 2MB，总包不超过 20MB
2. **审核规则**：各平台审核规则不同，提前了解避免被拒
3. **API 差异**：同一功能在不同平台 API 可能不同，使用条件编译
4. **登录态管理**：各平台登录机制不同，统一用后端 session 管理
5. **性能优化**：使用分包加载、按需注入、图片压缩等手段优化
