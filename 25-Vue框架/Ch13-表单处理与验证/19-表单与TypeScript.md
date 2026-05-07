# 表单与TypeScript

## 一、概念说明

TypeScript为表单提供类型安全：表单数据类型、验证规则类型、事件类型等。

```vue
<script setup lang="ts">
import { reactive, ref } from 'vue'

interface LoginForm {
  username: string
  password: string
  remember: boolean
}

const form = reactive<LoginForm>({
  username: '',
  password: '',
  remember: false
})

const errors = reactive<Partial<Record<keyof LoginForm, string>>>({})

const validate = (): boolean => {
  errors.username = form.username ? '' : '用户名必填'
  errors.password = form.password.length >= 6 ? '' : '密码至少6位'
  return !Object.values(errors).some(Boolean)
}

const handleSubmit = async () => {
  if (!validate()) return
  const res = await api.login(form)
}
</script>
```

## 二、具体用法

### 泛型表单组件

```vue
<script setup lang="ts" generic="T extends Record<string, any>">
interface Props {
  fields: { name: keyof T; label: string; type: string }[]
  initialValues: T
}

const props = defineProps<Props>()
const formData = reactive<T>({ ...props.initialValues })

defineExpose({
  getData: (): T => ({ ...formData })
})
</script>
```

### 验证规则类型

```ts
type ValidationRule = (value: any) => string | true

interface FieldConfig {
  name: string
  label: string
  rules: ValidationRule[]
}

const rules: Record<string, ValidationRule[]> = {
  email: [
    (v) => !!v || '邮箱必填',
    (v) => /^\S+@\S+\.\S+$/.test(v) || '邮箱格式不正确'
  ]
}
```

## 三、注意事项与常见陷阱

1. 使用接口定义表单数据结构
2. `Partial<T>`用于错误对象（部分字段可能无错误）
3. `keyof T`限制字段名必须是表单属性
4. 泛型组件提高表单组件的类型复用性
5. 事件处理函数需要正确的类型注解
