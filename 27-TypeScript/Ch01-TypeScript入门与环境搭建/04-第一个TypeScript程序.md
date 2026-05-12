# 第一个 TypeScript 程序

## 一、概念说明

本节将带你完成从创建文件到运行程序的完整流程，包括编写 TypeScript 代码、使用 `tsc` 编译器编译、以及运行生成的 JavaScript 文件。TypeScript 是 JavaScript 的超集，所有合法的 JS 代码都是合法的 TS 代码，但 TS 增加了类型系统，能在编译阶段捕获大量潜在错误。

## 二、具体用法

### 2.1 创建项目结构

```bash
# 项目目录结构
my-ts-app/
├── src/
│   └── index.ts      # TypeScript 源文件
├── dist/              # 编译输出目录
├── tsconfig.json      # 编译配置
└── package.json       # 项目配置
```

### 2.2 编写 TypeScript 代码

```typescript
// src/index.ts
interface Student {
  name: string;
  age: number;
  scores: number[];
}

function calculateAverage(student: Student): number {
  const total = student.scores.reduce((sum, score) => sum + score, 0);
  return total / student.scores.length;
}

function printReport(student: Student): void {
  const avg = calculateAverage(student);
  console.log(`学生: ${student.name}`);
  console.log(`年龄: ${student.age}`);
  console.log(`平均分: ${avg.toFixed(2)}`);
  console.log(`是否及格: ${avg >= 60 ? "是" : "否"}`);
}

// 使用
const student: Student = {
  name: "张三",
  age: 20,
  scores: [85, 92, 78, 90, 88]
};

printReport(student);
```

### 2.3 编译与运行

```bash
# 使用项目配置编译
npx tsc

# 运行编译后的 JavaScript
node dist/index.js
```

**输出：**
```
学生: 张三
年龄: 20
平均分: 86.60
是否及格: 是
```

### 2.4 使用 ts-node 直接运行

```bash
# 安装 ts-node（无需手动编译）
npm install -D ts-node

# 直接运行 TypeScript 文件
npx ts-node src/index.ts
```

### 2.5 与 JavaScript 的对比

```javascript
// JavaScript 版本 —— 没有类型检查
function calculateAverage(student) {
  // 运行时才发现 student 为 undefined
  const total = student.scores.reduce((sum, score) => sum + score, 0);
  return total / student.scores.length;
}

// 传入错误参数，JS 不会报错，运行时才崩溃
calculateAverage({ name: "张三", age: 20 }); // TypeError: Cannot read property 'reduce' of undefined
```

```typescript
// TypeScript 版本 —— 编译时就能发现错误
function calculateAverage(student: Student): number {
  const total = student.scores.reduce((sum, score) => sum + score, 0);
  return total / student.scores.length;
}

// 编译时报错：缺少 'scores' 属性
calculateAverage({ name: "张三", age: 20 }); // ❌ 编译错误
```

### 2.6 常见项目初始化流程

```bash
# 1. 初始化项目
mkdir my-ts-app && cd my-ts-app
npm init -y

# 2. 安装 TypeScript
npm install -D typescript

# 3. 生成 tsconfig.json
npx tsc --init

# 4. 创建源码目录
mkdir src
echo 'console.log("Hello TypeScript");' > src/index.ts

# 5. 编译并运行
npx tsc
node dist/index.js
```

## 三、注意事项与常见陷阱

1. **文件扩展名**：TypeScript 源文件使用 `.ts` 扩展名，React 项目使用 `.tsx`
2. **编译产物**：默认 `.ts` 编译为同目录的 `.js`，用 `outDir` 改变输出位置
3. **类型注解不输出**：编译后的 JS 不包含任何类型信息，类型系统是零运行时开销的
4. **`ts-node` 仅用于开发**：生产环境应使用编译后的 JS 文件
5. **`tsc` 不做打包**：`tsc` 只负责编译，不处理模块打包，生产项目需要 Vite/Webpack 等工具
6. **严格模式**：建议在 `tsconfig.json` 中开启 `"strict": true`，获得最完整的类型检查
