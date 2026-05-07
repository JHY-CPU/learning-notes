# React Hook Form 实战

## 1. 基础用法

### 1.1 安装与配置

```bash
npm install react-hook-form
# 可选：配合 zod 做验证
npm install @hookform/resolvers zod
```

### 1.2 最简示例

```jsx
import { useForm } from 'react-hook-form';

function SimpleForm() {
  const {
    register,     // 注册表单字段
    handleSubmit, // 包装提交处理
    formState: { errors, isSubmitting },
  } = useForm({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data) => {
    // data 已经是类型安全的对象
    console.log(data); // { email: '...', password: '...' }
    await fetch('/api/login', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <input
          {...register('email', {
            required: '邮箱不能为空',
            pattern: {
              value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
              message: '邮箱格式不正确',
            },
          })}
          placeholder="邮箱"
        />
        {errors.email && <span>{errors.email.message}</span>}
      </div>

      <div>
        <input
          type="password"
          {...register('password', {
            required: '密码不能为空',
            minLength: { value: 6, message: '至少 6 位' },
          })}
          placeholder="密码"
        />
        {errors.password && <span>{errors.password.message}</span>}
      </div>

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? '提交中...' : '登录'}
      </button>
    </form>
  );
}
```

**核心思想**：`register` 返回 `{ onChange, onBlur, ref, name }`，展开到 input 上即可，不需要手动管理 state。

---

## 2. register 详解

```jsx
const { register } = useForm();

// 基本用法：只注册，无验证
<input {...register('username')} />

// 带验证规则
<input {...register('username', {
  required: '不能为空',
  minLength: { value: 3, message: '至少 3 个字符' },
  maxLength: { value: 20, message: '不超过 20 个字符' },
  pattern: { value: /^[a-zA-Z0-9_]+$/, message: '格式不正确' },
  validate: (value) => value !== 'admin' || '该用户名不可用',
})} />

// 多个自定义验证函数
<input {...register('password', {
  validate: {
    hasUpperCase: (v) => /[A-Z]/.test(v) || '需包含大写字母',
    hasLowerCase: (v) => /[a-z]/.test(v) || '需包含小写字母',
    hasNumber: (v) => /[0-9]/.test(v) || '需包含数字',
  },
})} />

// 异步验证
<input {...register('email', {
  validate: async (value) => {
    const res = await fetch(`/api/check-email?email=${value}`);
    const { available } = await res.json();
    return available || '该邮箱已被注册';
  },
})} />
```

---

## 3. 常用表单元素

### 3.1 文本/数字/密码

```jsx
<input {...register('name', { required: true })} />
<input type="number" {...register('age', { valueAsNumber: true })} />
<input type="password" {...register('password')} />
<textarea {...register('bio', { maxLength: 500 })} />
```

### 3.2 Select

```jsx
<select {...register('city', { required: '请选择城市' })}>
  <option value="">请选择</option>
  <option value="beijing">北京</option>
  <option value="shanghai">上海</option>
</select>
```

### 3.3 Radio

```jsx
<label>
  <input type="radio" value="male" {...register('gender', { required: true })} />
  男
</label>
<label>
  <input type="radio" value="female" {...register('gender')} />
  女
</label>
{errors.gender && <span>请选择性别</span>}
```

### 3.4 Checkbox

```jsx
// 单个 checkbox → 值为 boolean
<input type="checkbox" {...register('agreed', { required: '请同意协议' })} />

// 多个 checkbox → 用数组
{['reading', 'sports', 'music'].map((hobby) => (
  <label key={hobby}>
    <input type="checkbox" value={hobby} {...register('hobbies')} />
    {hobby}
  </label>
))}
```

---

## 4. Controller（受控组件适配）

当使用第三方 UI 库（如 Ant Design、MUI）时，它们的组件不支持 `ref` 直接注册，需要用 `Controller`：

```jsx
import { useForm, Controller } from 'react-hook-form';
import { Select, DatePicker, Switch } from 'antd';

function ControlledForm() {
  const { control, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(console.log)}>
      {/* Select */}
      <Controller
        name="city"
        control={control}
        rules={{ required: '请选择城市' }}
        render={({ field, fieldState: { error } }) => (
          <div>
            <Select {...field} placeholder="选择城市">
              <Select.Option value="beijing">北京</Select.Option>
              <Select.Option value="shanghai">上海</Select.Option>
            </Select>
            {error && <span>{error.message}</span>}
          </div>
        )}
      />

      {/* DatePicker */}
      <Controller
        name="birthday"
        control={control}
        rules={{ required: '请选择日期' }}
        render={({ field }) => <DatePicker {...field} />}
      />

      {/* Switch */}
      <Controller
        name="newsletter"
        control={control}
        render={({ field: { value, onChange } }) => (
          <Switch checked={value} onChange={onChange} />
        )}
      />

      <button type="submit">提交</button>
    </form>
  );
}
```

---

## 5. 配合 Zod 验证

```jsx
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  username: z.string().min(3, '至少3个字符').max(20),
  email: z.string().email('邮箱格式不正确'),
  password: z.string().min(8, '至少8位'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: '两次密码不一致',
  path: ['confirmPassword'],
});

function ZodForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: zodResolver(schema),
  });

  return (
    <form onSubmit={handleSubmit(console.log)}>
      <input {...register('username')} placeholder="用户名" />
      {errors.username && <span>{errors.username.message}</span>}

      <input {...register('email')} placeholder="邮箱" />
      {errors.email && <span>{errors.email.message}</span>}

      <input type="password" {...register('password')} placeholder="密码" />
      {errors.password && <span>{errors.password.message}</span>}

      <input type="password" {...register('confirmPassword')} placeholder="确认密码" />
      {errors.confirmPassword && <span>{errors.confirmPassword.message}</span>}

      <button type="submit">注册</button>
    </form>
  );
}
```

---

## 6. useFieldArray 动态字段

```jsx
import { useForm, useFieldArray } from 'react-hook-form';

function DynamicForm() {
  const { register, control, handleSubmit } = useForm({
    defaultValues: {
      items: [{ name: '', quantity: 1, price: 0 }],
    },
  });

  const { fields, append, prepend, remove, swap, move, insert } = useFieldArray({
    control,
    name: 'items',
  });

  const onSubmit = (data) => {
    // data.items = [{ name: '...', quantity: 1, price: 10 }, ...]
    const total = data.items.reduce((sum, item) => sum + item.quantity * item.price, 0);
    console.log({ ...data, total });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {fields.map((field, index) => (
        <div key={field.id} style={{ display: 'flex', gap: 8 }}>
          <input
            {...register(`items.${index}.name`, { required: true })}
            placeholder="商品名"
          />
          <input
            type="number"
            {...register(`items.${index}.quantity`, { valueAsNumber: true, min: 1 })}
            placeholder="数量"
          />
          <input
            type="number"
            {...register(`items.${index}.price`, { valueAsNumber: true, min: 0 })}
            placeholder="单价"
          />
          <button type="button" onClick={() => remove(index)}>删除</button>
        </div>
      ))}
      <button type="button" onClick={() => append({ name: '', quantity: 1, price: 0 })}>
        + 添加商品
      </button>
      <button type="submit">结算</button>
    </form>
  );
}
```

---

## 7. 常用 API 速查

### 7.1 useForm 返回值

```jsx
const {
  register,          // 注册字段
  handleSubmit,      // 提交包装
  control,           // Controller/useFieldArray 需要
  watch,             // 监听字段值变化
  getValues,         // 获取当前值（不触发重渲染）
  setValue,          // 手动设置值
  reset,             // 重置表单
  resetField,        // 重置单个字段
  clearErrors,       // 清除错误
  setError,          // 手动设置错误
  trigger,           // 手动触发验证
  formState: {
    errors,          // 验证错误
    isDirty,         // 是否修改过
    dirtyFields,     // 哪些字段被修改了
    isSubmitted,     // 是否已提交
    isSubmitting,    // 是否正在提交
    isSubmitSuccessful, // 是否提交成功
    isValid,         // 是否全部验证通过
    isValidating,    // 是否正在异步验证
    submitCount,     // 提交次数
    touchedFields,   // 被触碰过的字段
  },
} = useForm({ defaultValues: {}, resolver: zodResolver(schema) });
```

### 7.2 watch 实时监听

```jsx
const { watch } = useForm();

// 监听单个字段
const password = watch('password');

// 监听多个字段
const [email, password] = watch(['email', 'password']);

// 监听所有字段
const allValues = watch();

// 带回调
useEffect(() => {
  const subscription = watch((value, { name, type }) => {
    console.log(`${name} changed to ${value[name]}`);
  });
  return () => subscription.unsubscribe();
}, [watch]);
```

### 7.3 setValue 手动赋值

```jsx
const { setValue } = useForm();

// 基本用法
setValue('username', '张三');

// 带验证选项
setValue('email', 'test@example.com', {
  shouldValidate: true,  // 设置后立即验证
  shouldDirty: true,     // 标记为已修改
  shouldTouch: true,     // 标记为已触碰
});

// 批量设置
setValue('username', '张三');
setValue('email', 'zhangsan@example.com');
```

---

## 8. 性能优化

```jsx
import { useForm, useFormState } from 'react-hook-form';

function OptimizedForm() {
  const { register, control, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(console.log)}>
      {/* 每个字段只在自己的值变化时重渲染 */}
      <input {...register('field1')} />
      <input {...register('field2')} />

      {/* 把 errors 提取到子组件，避免整个表单重渲染 */}
      <ErrorDisplay control={control} />

      <button type="submit">提交</button>
    </form>
  );
}

function ErrorDisplay({ control }) {
  // useFormState 只在 errors 变化时重渲染此组件
  const { errors } = useFormState({ control });

  return (
    <div>
      {Object.entries(errors).map(([key, err]) => (
        <p key={key} className="error">{err.message}</p>
      ))}
    </div>
  );
}
```

---

## 总结

| 功能 | API | 说明 |
|------|-----|------|
| 注册字段 | `register('name', rules)` | 自动绑定 ref/onChange/onBlur |
| 受控适配 | `Controller` | 第三方组件需要 |
| 动态列表 | `useFieldArray` | 增删改查字段数组 |
| 表单验证 | `zodResolver(schema)` | 声明式验证 |
| 实时监听 | `watch('name')` | 监听值变化 |
| 手动操作 | `setValue/reset/trigger` | 编程式控制 |
| 性能 | `useFormState` | 局部重渲染 |
