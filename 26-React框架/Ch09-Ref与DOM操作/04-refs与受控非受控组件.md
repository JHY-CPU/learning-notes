# Refs 与受控/非受控组件

React 中的表单有两种管理模式：受控组件和非受控组件。理解它们的区别和适用场景对构建表单至关重要。

---

## 一、受控组件 (Controlled Components)

受控组件的值由 React state 驱动，通过 `value` + `onChange` 模式管理。

### 1.1 基础模式

```jsx
function ControlledInput() {
  const [value, setValue] = useState('');

  return (
    <div>
      <input
        value={value}                    // 值由 state 控制
        onChange={(e) => setValue(e.target.value)}  // 变化时更新 state
      />
      <p>当前值: {value}</p>
      <button onClick={() => setValue('')}>清空</button>
    </div>
  );
}
```

### 1.2 工作流程

```
用户输入 "H"
  │
  ├── 1. onChange 被触发
  │
  ├── 2. 调用 setValue("H")
  │
  ├── 3. State 更新 → 触发重渲染
  │
  ├── 4. input 的 value 重新设为 "H"
  │
  └── 5. 用户看到 "H"
```

### 1.3 多输入表单

```jsx
function ControlledForm() {
  const [form, setForm] = useState({
    name: '',
    email: '',
    age: '',
    subscribe: false,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('提交:', form);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="name" value={form.name} onChange={handleChange} placeholder="姓名" />
      <input name="email" value={form.email} onChange={handleChange} placeholder="邮箱" />
      <input name="age" value={form.age} onChange={handleChange} type="number" />
      <label>
        <input name="subscribe" type="checkbox" checked={form.subscribe} onChange={handleChange} />
        订阅
      </label>
      <button type="submit">提交</button>
    </form>
  );
}
```

### 1.4 受控组件的优势

```jsx
// 即时验证
function ValidatedInput() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const value = e.target.value;
    setEmail(value);

    // 每次输入都验证
    if (value && !value.includes('@')) {
      setError('请输入有效的邮箱地址');
    } else {
      setError('');
    }
  };

  return (
    <div>
      <input value={email} onChange={handleChange} />
      {error && <span className="error">{error}</span>}
    </div>
  );
}

// 格式化输入
function PhoneInput() {
  const [phone, setPhone] = useState('');

  const handleChange = (e) => {
    // 自动格式化为 XXX-XXXX-XXXX
    const raw = e.target.value.replace(/\D/g, '').slice(0, 11);
    const formatted = raw.replace(/(\d{3})(\d{0,4})(\d{0,4})/, (_, a, b, c) =>
      [a, b, c].filter(Boolean).join('-')
    );
    setPhone(formatted);
  };

  return <input value={phone} onChange={handleChange} placeholder="手机号" />;
}
```

---

## 二、非受控组件 (Uncontrolled Components)

非受控组件的值由 DOM 自身管理，通过 ref 获取值。

### 2.1 基础模式

```jsx
function UncontrolledInput() {
  const inputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    // 通过 ref 读取值
    console.log('提交值:', inputRef.current.value);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input ref={inputRef} defaultValue="初始值" />
      <button type="submit">提交</button>
    </form>
  );
}
```

### 2.2 defaultValue vs value

```jsx
// 受控组件：value + onChange
<input value={name} onChange={e => setName(e.target.value)} />
// value 始终由 state 驱动，DOM 的 value 被 React 接管

// 非受控组件：defaultValue + ref
<input defaultValue="初始值" ref={inputRef} />
// defaultValue 只在首次渲染时设置，之后 DOM 自行管理
// 用户输入会直接改变 DOM 的 value，不经过 React
```

| 特性 | 受控 | 非受控 |
|---|---|---|
| 值的来源 | React state | DOM |
| 获取值 | 直接读 state | ref.current.value |
| 设置值 | setState | ref.current.value = x |
| 初始值 | `value={...}` | `defaultValue={...}` |
| 重渲染 | 每次输入都重渲染 | 输入不触发重渲染 |
| 验证时机 | onChange 中 | 提交时或 ref 读取时 |

---

## 三、文件输入的特殊性

文件输入 (`<input type="file">`) **必须**是非受控的。

```jsx
function FileUpload() {
  const fileInputRef = useRef(null);
  const [preview, setPreview] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    const files = fileInputRef.current.files;

    if (files.length === 0) {
      alert('请选择文件');
      return;
    }

    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));
    // 发送 formData...
  };

  const handleChange = () => {
    const file = fileInputRef.current.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={handleChange}
      />
      {preview && <img src={preview} alt="预览" />}
      <button type="submit">上传</button>
    </form>
  );
}
```

> React 18.2+ 支持受控的文件输入，但通常仍推荐非受控方式。

---

## 四、混合使用

实际项目中经常混合使用受控和非受控组件。

### 4.1 按需选择

```jsx
function MixedForm() {
  // 受控：需要即时验证和格式化
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');

  // 非受控：不需要实时读取值
  const bioRef = useRef(null);
  const fileRef = useRef(null);

  const validateEmail = (value) => {
    if (!value.includes('@')) {
      setEmailError('邮箱格式不正确');
    } else {
      setEmailError('');
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // 受控值直接使用 state
    console.log('邮箱:', email);
    // 非受控值通过 ref 获取
    console.log('简介:', bioRef.current.value);
    console.log('文件:', fileRef.current.files);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* 受控：即时验证 */}
      <input
        value={email}
        onChange={(e) => {
          setEmail(e.target.value);
          validateEmail(e.target.value);
        }}
      />
      {emailError && <span className="error">{emailError}</span>}

      {/* 非受控：纯文本区域 */}
      <textarea ref={bioRef} defaultValue="" rows={4} />

      {/* 非受控：文件输入 */}
      <input ref={fileRef} type="file" />

      <button type="submit">提交</button>
    </form>
  );
}
```

### 4.2 受控 + ref 获取 DOM

```jsx
function ControlledWithRef() {
  const [value, setValue] = useState('');
  const inputRef = useRef(null);

  // 受控管理值，ref 用于 DOM 操作
  const handleClear = () => {
    setValue('');
    inputRef.current.focus();  // 清空后重新聚焦
  };

  return (
    <div>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button onClick={handleClear}>清空并聚焦</button>
    </div>
  );
}
```

---

## 五、Form Ref 模式

使用 ref 获取整个表单的数据，而不为每个字段维护 state。

```jsx
function FormWithRef() {
  const formRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    // 通过 FormData API 获取所有字段值
    const formData = new FormData(formRef.current);
    const data = Object.fromEntries(formData.entries());

    // 复选框处理
    data.subscribe = formData.has('subscribe');

    console.log(data);
    // { name: "张三", email: "a@b.com", age: "25", subscribe: true }
  };

  return (
    <form ref={formRef} onSubmit={handleSubmit}>
      <input name="name" defaultValue="" placeholder="姓名" />
      <input name="email" type="email" placeholder="邮箱" />
      <input name="age" type="number" placeholder="年龄" />
      <label>
        <input name="subscribe" type="checkbox" defaultChecked />
        订阅新闻
      </label>
      <select name="city" defaultValue="beijing">
        <option value="beijing">北京</option>
        <option value="shanghai">上海</option>
      </select>
      <button type="submit">提交</button>
    </form>
  );
}
```

---

## 六、选择建议

### 决策流程

```
1. 需要实时验证或格式化？
   └── Yes → 受控

2. 需要根据输入值动态改变 UI？
   └── Yes → 受控

3. 需要在提交前读取值就够了？
   └── Yes → 非受控 + ref

4. 是文件上传？
   └── Yes → 非受控

5. 字段数量很多且不需要实时验证？
   └── Yes → 非受控 + FormData

6. 不确定？
   └── 受控（更灵活，React 推荐）
```

### 推荐

| 场景 | 推荐方式 |
|---|---|
| 登录/注册表单 | 受控 |
| 搜索框 | 受控 |
| 大型数据输入表单 | 非受控 + FormData |
| 文件上传 | 非受控 |
| 富文本编辑器 | 非受控（或特定库管理） |
| 第三方表单库 | 由库决定（通常混合） |
| 简单的筛选表单 | 非受控 + FormData |

### 表单库推荐

对于复杂表单，推荐使用专门的库：
- **react-hook-form**：高性能，非受控为主，受控也支持
- **formik**：受控为主，API 友好
- **zod + react-hook-form**：类型安全的验证
