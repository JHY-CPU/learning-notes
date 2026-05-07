# uni-app 网络与存储

## 一、概念说明

uni-app 提供了统一的网络请求和本地存储 API，跨平台兼容。

```javascript
// 网络请求封装
const request = (options) => {
  return new Promise((resolve, reject) => {
    uni.request({
      url: options.url,
      method: options.method || 'GET',
      data: options.data,
      header: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${uni.getStorageSync('token')}`
      },
      success: (res) => {
        if (res.statusCode === 200) resolve(res.data)
        else if (res.statusCode === 401) {
          uni.removeStorageSync('token')
          uni.reLaunch({ url: '/pages/login/login' })
          reject(new Error('未授权'))
        } else {
          reject(new Error(`请求失败: ${res.statusCode}`))
        }
      },
      fail: (err) => {
        uni.showToast({ title: '网络错误', icon: 'none' })
        reject(err)
      }
    })
  })
}

// 使用
const fetchData = async () => {
  const data = await request({
    url: 'https://api.example.com/data',
    method: 'POST',
    data: { key: 'value' }
  })
}
```

## 二、API 封装

```javascript
// utils/api.js
const BASE_URL = 'https://api.example.com'

class Api {
  get(url, params = {}) {
    return request({ url: BASE_URL + url, method: 'GET', data: params })
  }

  post(url, data = {}) {
    return request({ url: BASE_URL + url, method: 'POST', data })
  }

  put(url, data = {}) {
    return request({ url: BASE_URL + url, method: 'PUT', data })
  }

  delete(url) {
    return request({ url: BASE_URL + url, method: 'DELETE' })
  }

  // 文件上传
  upload(url, filePath) {
    return new Promise((resolve, reject) => {
      uni.uploadFile({
        url: BASE_URL + url,
        filePath,
        name: 'file',
        header: { 'Authorization': `Bearer ${uni.getStorageSync('token')}` },
        success: (res) => resolve(JSON.parse(res.data)),
        fail: reject
      })
    })
  }
}

export default new Api()

// services/user.js
import api from '@/utils/api'

export const userService = {
  login: (data) => api.post('/auth/login', data),
  getInfo: () => api.get('/user/info'),
  update: (data) => api.put('/user/info', data),
  uploadAvatar: (filePath) => api.upload('/user/avatar', filePath)
}
```

## 三、本地存储

```javascript
// 同步存储
uni.setStorageSync('key', 'value')
const value = uni.getStorageSync('key')
uni.removeStorageSync('key')
uni.clearStorageSync()

// 异步存储
uni.setStorage({
  key: 'userInfo',
  data: { name: '张三', age: 25 },
  success: () => console.log('存储成功')
})

uni.getStorage({
  key: 'userInfo',
  success: (res) => console.log(res.data)
})

// 封装存储工具
const storage = {
  set(key, value, expireMinutes = null) {
    const data = {
      value,
      expire: expireMinutes ? Date.now() + expireMinutes * 60000 : null
    }
    uni.setStorageSync(key, JSON.stringify(data))
  },
  get(key) {
    try {
      const raw = uni.getStorageSync(key)
      if (!raw) return null
      const data = JSON.parse(raw)
      if (data.expire && Date.now() > data.expire) {
        uni.removeStorageSync(key)
        return null
      }
      return data.value
    } catch {
      return null
    }
  },
  remove(key) {
    uni.removeStorageSync(key)
  }
}
```

## 四、文件操作

```javascript
// 选择图片
const chooseImage = async () => {
  const res = await uni.chooseImage({
    count: 1,
    sizeType: ['compressed'],
    sourceType: ['album', 'camera']
  })
  return res.tempFilePaths[0]
}

// 保存文件
const saveFile = async (tempFilePath) => {
  const res = await uni.saveFile({ tempFilePath })
  return res.savedFilePath
}

// 获取文件信息
const getFileInfo = async (filePath) => {
  const res = await uni.getFileInfo({ filePath })
  return res
}
```

## 五、注意事项与常见陷阱

1. **HTTPS 要求**：各平台要求使用 HTTPS，开发时可在设置中关闭校验
2. **请求并发限制**：微信小程序同时最多 10 个请求
3. **存储容量限制**：各平台本地存储容量不同，大量数据建议用 SQLite
4. **文件路径**：选择的文件路径是临时路径，需要保存才能持久化
5. **跨域问题**：H5 端需要配置代理解决跨域
