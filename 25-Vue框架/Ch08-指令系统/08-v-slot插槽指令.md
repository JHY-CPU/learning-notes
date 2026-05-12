# v-slot 插槽指令

## 一、概念说明
`v-slot` 用于向子组件的**插槽传递内容**。缩写为 `#`。支持默认插槽、具名插槽和作用域插槽。

## 二、具体用法

### 2.1 默认插槽
```vue
<template>
  <!-- 完整写法 -->
  <MyComponent v-slot:default>
    <p>默认插槽内容</p>
  </MyComponent>

  <!-- 简写（默认插槽可省略 default） -->
  <MyComponent>
    <p>默认插槽内容</p>
  </MyComponent>
</template>
```

### 2.2 具名插槽
```vue
<template>
  <Layout>
    <template v-slot:header>
      <h1>页面标题</h1>
    </template>

    <template v-slot:default>
      <p>主内容区</p>
    </template>

    <template v-slot:footer>
      <p>页脚</p>
    </template>
  </Layout>
</template>
```

### 2.3 作用域插槽
```vue
<template>
  <UserList :users="list">
    <!-- 接收子组件传递的数据 -->
    <template v-slot:default="{ user, index }">
      <span>{{ index + 1 }}. {{ user.name }}</span>
    </template>
  </UserList>
</template>
```

### 2.4 缩写语法
```vue
<template>
  <!-- # 替代 v-slot -->
  <Layout>
    <template #header>标题</template>
    <template #default>内容</template>
    <template #footer="{ year }">版权 {{ year }}</template>
  </Layout>
</template>
```

### 2.5 动态插槽名
```vue
<template>
  <Layout>
    <template #[slotName]>
      <p>动态插槽内容</p>
    </template>
  </Layout>
</template>
<script setup>
import { ref } from 'vue'
const slotName = ref('header')
</script>
```

## 三、注意事项与常见陷阱
- `v-slot` 只能用在 `<template>` 或组件上
- 默认插槽的内容不能同时使用缩写和具名插槽语法
- 动态插槽名用方括号：`#[dynamicName]`
- 作用域插槽使子组件成为"渲染器"

## 四、v-slot 的完整语法

```
v-slot:slotName="slotProps"
  │        │         │
  │        │         └── 插槽作用域数据（可解构）
  │        └── 插槽名
  └── 指令

简写: #slotName="slotProps"
```

## 五、实际场景

### 5.1 数据表格组件
```vue
<!-- Table.vue -->
<template>
  <table>
    <thead>
      <tr>
        <th v-for="col in columns" :key="col.key">{{ col.label }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(row, rowIndex) in data" :key="row.id">
        <td v-for="col in columns" :key="col.key">
          <slot :name="`cell-${col.key}`" :row="row" :value="row[col.key]" :index="rowIndex">
            {{ row[col.key] }}  <!-- 默认渲染 -->
          </slot>
        </td>
      </tr>
    </tbody>
  </table>
</template>
```

```vue
<!-- 使用 -->
<template>
  <Table :columns="cols" :data="users">
    <template #cell-name="{ row, value }">
      <strong>{{ value }}</strong>
    </template>
    <template #cell-status="{ value }">
      <span :class="`status-${value}`">{{ statusLabels[value] }}</span>
    </template>
    <!-- 其他列使用默认渲染 -->
  </Table>
</template>
```

### 5.2 条件插槽
```vue
<template>
  <Card>
    <template v-if="showHeader" #header>
      <h2>{{ title }}</h2>
    </template>
    <template #default>
      <p>内容</p>
    </template>
  </Card>
</template>
```
