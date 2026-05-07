# UI组件库Naive-UI

## 一、概念说明

Naive UI 是 TypeScript 优先的 Vue 3 UI 组件库，由图森未来开源。它的类型支持是所有 Vue UI 库中最好的，组件 API 设计简洁一致。全量使用仅增加约 12MB 打包体积（Tree Shaking 后更小），性能优秀。

## 二、具体用法

### 安装

```bash
npm install naive-ui
```

```vue
<!-- App.vue -->
<script setup lang="ts">
import {
  NConfigProvider,
  NMessageProvider,
  NDialogProvider,
  darkTheme,
  type GlobalThemeOverrides
} from 'naive-ui'

const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#18a058',
    primaryColorHover: '#36ad6a'
  }
}
</script>

<template>
  <NConfigProvider :theme-overrides="themeOverrides">
    <NMessageProvider>
      <NDialogProvider>
        <router-view />
      </NDialogProvider>
    </NMessageProvider>
  </NConfigProvider>
</template>
```

### 组件使用

```vue
<script setup lang="ts">
import {
  NButton,
  NInput,
  NCard,
  NSpace,
  useMessage,
  useDialog
} from 'naive-ui'

const message = useMessage()
const dialog = useDialog()

const inputVal = ref('')

function handleSuccess() {
  message.success('操作成功！')
  // 页面右上角弹出绿色成功提示
}

function handleConfirm() {
  dialog.warning({
    title: '确认删除',
    content: '确定要删除这条记录吗？',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: () => {
      message.success('已删除')
    }
  })
}
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="Naive UI 示例">
      <NSpace vertical>
        <NInput v-model:value="inputVal" placeholder="请输入内容" />
        <NSpace>
          <NButton type="primary" @click="handleSuccess">成功提示</NButton>
          <NButton type="warning" @click="handleConfirm">确认对话框</NButton>
        </NSpace>
      </NSpace>
    </NCard>

    <!-- 数据表格 -->
    <NDataTable
      :columns="[
        { title: '姓名', key: 'name' },
        { title: '年龄', key: 'age' },
        { title: '地址', key: 'address' }
      ]"
      :data="[
        { name: '张三', age: 25, address: '北京' },
        { name: '李四', age: 30, address: '上海' }
      ]"
    />
  </NSpace>
</template>
```

### TypeScript 优势

```vue
<script setup lang="ts">
import type { DataTableColumns, SelectOption } from 'naive-ui'

// 列定义有完整类型
const columns: DataTableColumns = [
  {
    title: '状态',
    key: 'status',
    render(row) {
      // row 类型完整推断
      return h('span', row.status)
    }
  }
]

// 选项有类型约束
const options: SelectOption[] = [
  { label: '选项A', value: 'a' },
  { label: '选项B', value: 'b', disabled: true }
]
</script>
```

## 三、注意事项与常见陷阱

1. **需要手动导入组件**：不支持全局全量注册（设计如此，利于 Tree Shaking）
2. **useMessage 需要在 NMessageProvider 内部调用**：否则报错
3. **主题覆盖用 GlobalThemeOverrides**：不是 CSS 变量方式
4. **日期组件依赖 date-fns**：不是 dayjs
5. **中英文文档都很完善**：遇到问题优先查看官方文档
