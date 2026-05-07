# UI组件库Ant-Design-Vue

## 一、概念说明

Ant Design Vue 是 Ant Design 的 Vue 实现，由社区维护，提供 60+ 高质量组件。设计语言来自蚂蚁集团，适合企业级中后台应用。Ant Design Vue 4.x 支持 Vue 3，API 设计与 React 版 Ant Design 保持一致。

## 二、具体用法

### 安装

```bash
npm install ant-design-vue
```

```ts
// main.ts - 全量引入
import { createApp } from 'vue'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import App from './App.vue'

createApp(App).use(Antd).mount('#app')
```

### 按需引入

```ts
// 使用 unplugin-vue-components 自动按需引入
// vite.config.ts
import Components from 'unplugin-vue-components/vite'
import { AntDesignVueResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    Components({
      resolvers: [AntDesignVueResolver({ importStyle: false })]
    })
  ]
})
```

### 表格与表单

```vue
<script setup lang="ts">
import { message } from 'ant-design-vue'
import type { TableColumnsType } from 'ant-design-vue'

const columns: TableColumnsType = [
  { title: '姓名', dataIndex: 'name', key: 'name' },
  { title: '年龄', dataIndex: 'age', key: 'age', sorter: (a, b) => a.age - b.age },
  { title: '地址', dataIndex: 'address', key: 'address' },
  {
    title: '操作',
    key: 'action',
    customRender: ({ record }) => {
      return h('a', { onClick: () => handleEdit(record) }, '编辑')
    }
  }
]

const dataSource = ref([
  { key: '1', name: '张三', age: 25, address: '北京市朝阳区' },
  { key: '2', name: '李四', age: 30, address: '上海市浦东新区' }
])

const formState = reactive({ name: '', email: '' })

function handleEdit(record: any) {
  message.info(`编辑用户: ${record.name}`)
  // 页面顶部弹出蓝色信息提示
}

function handleSubmit() {
  if (!formState.name) {
    message.error('请输入姓名')
    return
  }
  message.success('提交成功')
}
</script>

<template>
  <div>
    <a-form :model="formState" layout="inline">
      <a-form-item label="姓名">
        <a-input v-model:value="formState.name" placeholder="请输入" />
      </a-form-item>
      <a-form-item label="邮箱">
        <a-input v-model:value="formState.email" placeholder="请输入" />
      </a-form-item>
      <a-form-item>
        <a-button type="primary" @click="handleSubmit">提交</a-button>
      </a-form-item>
    </a-form>

    <a-table :columns="columns" :data-source="dataSource" :pagination="{ pageSize: 5 }" />
  </div>
</template>
```

### 响应式栅格

```vue
<template>
  <a-row :gutter="[16, 16]">
    <a-col :xs="24" :sm="12" :md="8" :lg="6">
      <a-card title="统计卡片">内容1</a-card>
    </a-col>
    <a-col :xs="24" :sm="12" :md="8" :lg="6">
      <a-card title="统计卡片">内容2</a-card>
    </a-col>
    <a-col :xs="24" :sm="12" :md="8" :lg="6">
      <a-card title="统计卡片">内容3</a-card>
    </a-col>
    <a-col :xs="24" :sm="12" :md="8" :lg="6">
      <a-card title="统计卡片">内容4</a-card>
    </a-col>
  </a-row>
</template>
```

## 三、注意事项与常见陷阱

1. **组件名带 a- 前缀**：`<a-button>`、`<a-table>`，与 React 版不同
2. **message/Modal 等是函数调用**：`message.success()` 而非组件
3. **日期组件依赖 dayjs**：需要单独安装 `npm install dayjs`
4. **Form 的 rules 与 Element Plus 不同**：使用 `rules` 数组格式
5. **Vue 3 请使用 4.x 版本**：3.x 不支持 Vue 3
