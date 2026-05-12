# POST 请求详解

## 一、概念说明

POST 请求用于向服务器提交数据，数据放在请求体（body）中。常用于创建资源、表单提交、文件上传等场景。

```vue
<script setup>
import { ref } from 'vue'
import axios from 'axios'

const form = ref({ name: '', email: '', age: 18 })
const result = ref(null)

async function submitForm() {
  const { data } = await axios.post('/api/users', form.value)
  result.value = data
}
</script>

<template>
  <form @submit.prevent="submitForm">
    <input v-model="form.name" placeholder="姓名" />
    <input v-model="form.email" placeholder="邮箱" />
    <input v-model.number="form.age" type="number" placeholder="年龄" />
    <button type="submit">提交</button>
  </form>
  <pre v-if="result">{{ JSON.stringify(result, null, 2) }}</pre>
</template>
```

## 二、具体用法

### 2.1 JSON 格式提交

```js
// Content-Type: application/json（默认）
axios.post('/api/users', { name: '张三', age: 25 })
```

### 2.2 FormData 格式提交

```js
const formData = new FormData()
formData.append('name', '张三')
formData.append('avatar', fileInput.files[0])

axios.post('/api/users', formData)
// axios 自动设置 Content-Type: multipart/form-data
```

### 2.3 URL 编码格式

```js
const params = new URLSearchParams()
params.append('username', 'admin')
params.append('password', '123456')

axios.post('/api/login', params)
// Content-Type: application/x-www-form-urlencoded
```

### 2.4 区别对比

| 格式 | Content-Type | 适用场景 |
|------|-------------|---------|
| JSON | `application/json` | RESTful API |
| FormData | `multipart/form-data` | 文件上传 |
| URL 编码 | `application/x-www-form-urlencoded` | 传统表单 |

## 四、文件上传

```vue
<script setup>
import { ref } from 'vue'
import axios from 'axios'

const file = ref(null)
const progress = ref(0)
const uploading = ref(false)

async function uploadFile() {
  if (!file.value) return

  const formData = new FormData()
  formData.append('file', file.value)

  uploading.value = true
  try {
    const { data } = await axios.post('/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (e.total) {
          progress.value = Math.round((e.loaded / e.total) * 100)
        }
      }
    })
    console.log('上传成功:', data)
  } finally {
    uploading.value = false
    progress.value = 0
  }
}
</script>

<template>
  <input type="file" @change="e => file = e.target.files[0]" />
  <button @click="uploadFile" :disabled="uploading || !file">
    {{ uploading ? `上传中 ${progress}%` : '上传' }}
  </button>
  <div v-if="uploading" class="progress-bar">
    <div :style="{ width: progress + '%' }"></div>
  </div>
</template>
```

## 五、批量提交

```js
// 方式1：数组格式
await axios.post('/api/users/batch', {
  users: [
    { name: '张三', age: 25 },
    { name: '李四', age: 30 }
  ]
})

// 方式2：逐个提交（并行）
const results = await Promise.all(
  users.map(user => axios.post('/api/users', user))
)

// 方式3：逐个提交（串行，含进度）
for (let i = 0; i < users.length; i++) {
  await axios.post('/api/users', users[i])
  console.log(`进度: ${i + 1}/${users.length}`)
}
```

## 三、注意事项与常见陷阱

- 后端接口要求的 Content-Type 格式要与前端发送的一致
- 上传文件时不要手动设置 `Content-Type`，让浏览器自动生成 boundary
- POST 请求的响应体通常包含创建后的资源数据
- 文件上传的进度回调只在 XMLHttpRequest 方式下可用（axios 默认使用）
- `Promise.all` 批量提交时，一个失败会导致全部失败
