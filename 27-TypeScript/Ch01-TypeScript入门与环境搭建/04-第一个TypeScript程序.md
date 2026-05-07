# 第一个 TypeScript 程序

## 一、概念说明

本节将带你完成从创建文件到运行程序的完整流程，包括编写 TypeScript 代码、使用 `tsc` 编译器编译、以及运行生成的 JavaScript 文件。

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

## 三、注意事项与常见陷阱

1. **文件扩展名**：TypeScript 源文件使用 `.ts` 扩展名
2. **编译产物**：默认 `.ts` 编译为同目录的 `.js`，用 `outDir` 改变输出位置
3. **类型注解不输出**：编译后的 JS 不包含任何类型信息
4. **`ts-node` 仅用于开发**：生产环境应使用编译后的 JS 文件
