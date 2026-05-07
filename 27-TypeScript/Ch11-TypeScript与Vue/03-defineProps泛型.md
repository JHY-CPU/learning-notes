# defineProps泛型

## 一、概念说明

`defineProps` 是 Vue 3 `<script setup>` 中定义组件 Props 的宏。使用泛型语法可以简洁地声明 Props 类型，TypeScript 会自动推断并提供完整的类型检查和 IDE 支持。

## 二、具体用法

### 2.1 基本泛型 Props

```vue
<script setup lang="ts">
// 使用泛型语法定义 Props
const props = defineProps<{
  title: string;
  count: number;
  disabled?: boolean;    // 可选属性
  items?: string[];      // 可选数组
}>();

// props 自动有正确的类型
console.log(props.title);   // string
console.log(props.count);   // number
console.log(props.disabled); // boolean | undefined
</script>
```

### 2.2 使用 interface 定义

```vue
<script setup lang="ts">
interface Props {
  title: string;
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
});

// props.variant 类型是 'primary' | 'secondary' | 'danger'
// props.size 类型是 'sm' | 'md' | 'lg'
</script>
```

### 2.3 withDefaults 设置默认值

```vue
<script setup lang="ts">
interface Props {
  title: string;
  message?: string;
  count?: number;
  visible?: boolean;
  items?: string[];
}

// withDefaults 的第二个参数设置默认值
const props = withDefaults(defineProps<Props>(), {
  message: '默认消息',
  count: 0,
  visible: true,
  items: () => [],   // 复杂默认值用工厂函数
});

// props.message 类型是 string（不是 string | undefined）
// props.count 类型是 number（不是 number | undefined）
</script>
```

### 2.4 泛型 Props

```vue
<script setup lang="ts">
// 泛型组件 — 处理多种数据类型
interface Props<T> {
  data: T[];
  keyField: keyof T;
  labelField: keyof T;
  onSelect?: (item: T) => void;
}

// 注意：defineProps 不直接支持泛型参数
// 需要用具体类型或运行时声明
interface Item {
  id: number;
  label: string;
}

const props = defineProps<Props<Item>>();
```

### 2.5 运行时 Props 声明（混合模式）

```vue
<script setup lang="ts">
import type { PropType } from 'vue';

// 运行时声明 + 类型标注
const props = defineProps({
  title: {
    type: String,
    required: true,
  },
  config: {
    type: Object as PropType<{ url: string; method: string }>,
    required: true,
  },
  items: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
});
</script>
```

## 三、注意事项与常见陷阱

1. **`defineProps` 的泛型语法是编译时宏**：不要在普通 TypeScript 代码中使用
2. **`withDefaults` 只能与泛型语法一起使用**：不能与运行时声明混用
3. **默认值的类型要严格匹配**：`count?: number` 的默认值必须是 `number`
4. **可选属性默认值由 `withDefaults` 提供**：访问时类型不是 `T | undefined`
5. **Props 类型不要用 `type` 关键字声明**：`defineProps` 不支持 `type` 别名语法
