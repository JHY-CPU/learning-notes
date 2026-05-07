# UI组件库Element-Plus

## 一、概念说明

Element Plus 是 Element UI 的 Vue 3 版本，基于 TypeScript 构建，提供 70+ 组件。设计风格简洁现代，是中文社区最流行的 Vue UI 库，尤其适合中后台管理系统。

## 二、具体用法

### 安装与全量引入

```bash
npm install element-plus @element-plus/icons-vue
```

```ts
// main.ts
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import App from './App.vue'

const app = createApp(App)
app.use(ElementPlus)

// 注册所有图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}
app.mount('#app')
```

### 按需引入

```ts
// vite.config.ts
import { defineConfig } from 'vite'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    AutoImport({
      resolvers: [ElementPlusResolver()]
    }),
    Components({
      resolvers: [ElementPlusResolver()]
    })
  ]
})
```

```vue
<!-- 按需引入后直接使用，无需 import -->
<template>
  <div>
    <el-button type="primary" @click="visible = true">打开对话框</el-button>
    <el-dialog v-model="visible" title="提示">
      <p>这是一个按需引入的对话框</p>
      <template #footer>
        <el-button @click="visible = false">取消</el-button>
        <el-button type="primary" @click="visible = false">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
const visible = ref(false)
</script>
```

### 表单与表格

```vue
<script setup lang="ts">
import { ref, reactive } from 'vue'
import type { FormInstance, FormRules } from 'element-plus'

const formRef = ref<FormInstance | null>(null)
const form = reactive({
  name: '',
  email: '',
  region: ''
})

const rules: FormRules = {
  name: [{ required: true, message: '请输入姓名', trigger: 'blur' }],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '邮箱格式不正确', trigger: 'blur' }
  ]
}

async function submitForm() {
  const valid = await formRef.value?.validate()
  if (valid) {
    console.log('表单数据:', form)
    // 输出：表单数据: { name: "张三", email: "zhang@example.com", region: "" }
  }
}
</script>

<template>
  <el-form ref="formRef" :model="form" :rules="rules" label-width="80px">
    <el-form-item label="姓名" prop="name">
      <el-input v-model="form.name" />
    </el-form-item>
    <el-form-item label="邮箱" prop="email">
      <el-input v-model="form.email" />
    </el-form-item>
    <el-form-item>
      <el-button type="primary" @click="submitForm">提交</el-button>
    </el-form-item>
  </el-form>
</template>
```

## 三、注意事项与常见陷阱

1. **全量引入增加包体积**：生产环境务必使用按需引入
2. **国际化需手动配置**：默认英文，中文需 `app.use(ElementPlus, { locale: zhCn })`
3. **暗色主题支持**：Element Plus 内置 dark mode，通过 CSS 变量切换
4. **表单验证是异步的**：`validate()` 返回 Promise
5. **图标需单独安装**：`@element-plus/icons-vue` 是独立包
