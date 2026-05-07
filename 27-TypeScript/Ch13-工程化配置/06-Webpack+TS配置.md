# Webpack+TS配置

## 一、概念说明

Webpack 通过 `ts-loader` 或 `babel-loader` + `@babel/preset-typescript` 支持 TypeScript。`ts-loader` 提供完整的类型检查，而 Babel 方案速度更快但不做类型检查。

## 二、具体用法

### 2.1 ts-loader 方案

```bash
npm install -D webpack ts-loader typescript
```

```javascript
// webpack.config.js
const path = require('path');

module.exports = {
  entry: './src/index.ts',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js'],
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
};
```

### 2.2 Babel 方案

```bash
npm install -D @babel/core @babel/preset-env @babel/preset-typescript babel-loader
```

```javascript
// babel.config.js
module.exports = {
  presets: [
    '@babel/preset-env',
    '@babel/preset-typescript',
  ],
};

// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'babel-loader',
        exclude: /node_modules/,
      },
    ],
  },
};
```

### 2.3 ts-loader 优化

```javascript
// 使用 fork-ts-checker-webpack-plugin 在单独进程中检查类型
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');

module.exports = {
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true, // 只转译，不检查类型
          },
        },
      },
    ],
  },
  plugins: [
    new ForkTsCheckerWebpackPlugin(), // 单独进程检查类型
  ],
};
```

### 2.4 tsconfig 配置

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "node",
    "jsx": "react-jsx",
    "sourceMap": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

## 三、注意事项与常见陷阱

1. **`ts-loader` 比 Babel 慢**：用 `transpileOnly` + ForkTsCheckerPlugin 优化
2. **Babel 不做类型检查**：需要 `tsc --noEmit` 单独检查
3. **source map 需要同时配置 Webpack 和 tsconfig**
4. **路径别名需要同时配置 Webpack 和 tsconfig**
5. **Webpack 5 支持 TS 配置**：`webpack.config.ts` 需要 `ts-node`
