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

## 四、完整类型化表单示例

```vue
<script setup lang="ts">
import { reactive, computed } from 'vue'

// 定义表单接口
interface RegistrationForm {
  username: string
  email: string
  password: string
  confirmPassword: string
  role: 'user' | 'admin' | 'editor'
  agreeTerms: boolean
}

// 错误类型
type FormErrors = Partial<Record<keyof RegistrationForm, string>>

const form = reactive<RegistrationForm>({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
  role: 'user',
  agreeTerms: false
})

const errors = reactive<FormErrors>({})

// 验证函数的类型
type Validator = (value: any, form?: RegistrationForm) => string | true

const validators: Record<keyof RegistrationForm, Validator[]> = {
  username: [
    (v: string) => !!v || '用户名必填',
    (v: string) => v.length >= 3 || '至少3个字符'
  ],
  email: [
    (v: string) => !!v || '邮箱必填',
    (v: string) => /\S+@\S+\.\S+/.test(v) || '邮箱格式不正确'
  ],
  password: [
    (v: string) => !!v || '密码必填',
    (v: string) => v.length >= 8 || '至少8位'
  ],
  confirmPassword: [
    (v: string, f?: RegistrationForm) => v === f?.password || '密码不一致'
  ],
  role: [],
  agreeTerms: [
    (v: boolean) => v || '必须同意条款'
  ]
}

function validateField(field: keyof RegistrationForm): boolean {
  const rules = validators[field]
  for (const rule of rules) {
    const result = rule(form[field], form)
    if (result !== true) {
      errors[field] = result
      return false
    }
  }
  delete errors[field]
  return true
}

const isValid = computed(() =>
  Object.keys(validators).every(field => validateField(field as keyof RegistrationForm))
)
</script>
```

## 五、表单状态管理类型

```ts
// 表单状态枚举
type FormStatus = 'idle' | 'validating' | 'submitting' | 'success' | 'error'

// 提交结果类型
interface SubmitResult<T> {
  success: boolean
  data?: T
  errors?: FormErrors
  message?: string
}

// 表单配置类型
interface FormFieldConfig<T = any> {
  name: keyof T
  label: string
  type: 'text' | 'email' | 'password' | 'select' | 'checkbox'
  placeholder?: string
  required?: boolean
  options?: { label: string; value: string | number }[]
  validators?: Validator[]
}

// 使用泛型表单Hook
function useForm<T extends Record<string, any>>(
  initialValues: T,
  fieldConfigs: FormFieldConfig<T>[]
) {
  const form = reactive<T>({ ...initialValues })
  const errors = reactive<Partial<Record<keyof T, string>>>({})
  const status = ref<FormStatus>('idle')

  const validate = (): boolean => { /* ... */ return true }
  const submit = async (handler: (data: T) => Promise<void>): Promise<void> => { /* ... */ }
  const reset = () => { Object.assign(form, initialValues) }

  return { form, errors, status, validate, submit, reset }
}
```

## 三、注意事项与常见陷阱

1. 使用接口定义表单数据结构
2. `Partial<T>`用于错误对象（部分字段可能无错误）
3. `keyof T`限制字段名必须是表单属性
4. 泛型组件提高表单组件的类型复用性
5. 事件处理函数需要正确的类型注解
6. 验证函数数组的类型签名要统一（返回`string | true`）
7. `reactive`对象的类型推断依赖初始值的类型
