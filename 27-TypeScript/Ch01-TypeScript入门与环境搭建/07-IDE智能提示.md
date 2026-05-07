# IDE 智能提示

## 一、概念说明

TypeScript 最直观的好处之一就是在编辑器中获得精确的智能提示（IntelliSense）。VS Code 内置了 TypeScript 语言服务，能够提供**自动补全**、**类型信息悬浮**、**错误高亮**、**跳转定义**等功能，极大提升开发效率。

## 二、具体用法

### 2.1 自动补全

```typescript
interface API {
  getUsers(): Promise<User[]>;
  createUser(user: Omit<User, "id">): Promise<User>;
  deleteUser(id: number): Promise<void>;
}

// 输入 api. 后，IDE 会列出所有可用方法
async function handleApi(api: API) {
  const users = await api.getUsers(); // 自动补全方法名
  console.log(users.length);
}
```

**输出：**
```
输入 "api." 后弹出补全列表：
- getUsers(): Promise<User[]>
- createUser(user: Omit<User, "id">): Promise<User>
- deleteUser(id: number): Promise<void>
```

### 2.2 类型信息悬浮

```typescript
// 鼠标悬停在变量上显示类型信息
const config = {
  port: 3000,
  host: "localhost",
  debug: true
};
// 悬停显示: const config: { port: number; host: string; debug: boolean; }
```

### 2.3 错误实时高亮

```typescript
// VS Code 会在输入时实时显示类型错误

// ❌ 红色波浪线提示错误
// const num: number = "hello";
// 提示: Type 'string' is not assignable to type 'number'.

// ✅ 修正后波浪线消失
const num: number = 42;
```

### 2.4 快捷操作

```typescript
// F12: 跳转到定义
// Shift+F12: 查找所有引用
// F2: 重命名符号（安全重构）
// Ctrl+Space: 手动触发补全
// Ctrl+Shift+P > "TypeScript: Restart TS Server": 重启语言服务
```

## 三、注意事项与常见陷阱

1. **TS Server 版本**：VS Code 可使用工作区 TypeScript 版本（状态栏右下角切换）
2. **大型项目性能**：项目过大时可在 `tsconfig.json` 中适当 `exclude` 无关目录
3. **路径映射**：配置 `paths` 后 IDE 能正确解析模块别名的自动补全
4. **`.js` 文件支持**：在 `jsconfig.json` 或 `tsconfig.json` 的 `allowJs` 中开启
