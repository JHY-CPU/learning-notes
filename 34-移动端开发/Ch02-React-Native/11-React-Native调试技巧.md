# React Native 调试技巧

## 一、概念说明

高效的调试能力是 React Native 开发者的核心技能。React Native 提供了多种调试工具和方法，包括内置开发者菜单、Chrome DevTools、Flipper 等。

```javascript
// 调试工具概览
/*
1. React Native 开发者菜单 (摇晃设备或 Cmd+D / Cmd+M)
2. Chrome DevTools (远程 JS 调试)
3. Flipper (Facebook 官方调试工具)
4. React DevTools (组件树和状态检查)
5. 日志输出 (console.log / console.warn / console.error)
6. 断点调试 (VS Code / Chrome)
*/
```

## 二、调试方法详解

### 2.1 Chrome DevTools

```javascript
// 启用远程调试
// 1. 打开开发者菜单 (摇晃设备)
// 2. 选择 "Debug JS Remotely"
// 3. Chrome 中打开 chrome://inspect

// 条件断点调试
const complexFunction = (data) => {
  debugger; // 代码断点，远程调试时会暂停
  const result = data
    .filter(item => item.active)
    .map(item => ({
      ...item,
      processed: true,
    }));
  return result;
};

// 调试异步代码
const asyncDebugExample = async () => {
  console.log('开始请求');
  try {
    const response = await fetch(url);
    console.log('响应状态:', response.status);
    const data = await response.json();
    console.log('数据:', JSON.stringify(data, null, 2));
    return data;
  } catch (error) {
    console.error('请求失败:', error);
    console.error('错误堆栈:', error.stack);
  }
};
```

### 2.2 Flipper 调试

```javascript
// Flipper 集成
// npm install react-native-flipper

import { logger } from 'react-native-logs';

const log = logger.createLogger({
  severity: __DEV__ ? logger.levels.DEBUG : logger.levels.ERROR,
  transport: __DEV__ ? [
    logger.consoleTransport,
    logger.fileTransport,
  ] : [logger.fileTransport],
  transportOptions: {
    colors: {
      info: 'blueBright',
      warn: 'yellowBright',
      error: 'redBright',
    },
  },
});

// 使用
log.debug('调试信息', { key: 'value' });
log.info('一般信息');
log.warn('警告信息');
log.error('错误信息', error);

// 网络请求监控 (集成 Flipper)
// 自动捕获所有 fetch 和 XMLHttpRequest 请求
```

### 2.3 React DevTools

```javascript
// 安装 React DevTools
// npm install -g react-devtools
// 然后运行: react-devtools

// 组件性能分析
const ProfiledComponent = () => {
  console.log('组件渲染');
  return <View>...</View>;
};

// 使用 Profiler API
import { Profiler } from 'react';

const onRenderCallback = (
  id,
  phase,
  actualDuration,
  baseDuration,
  startTime,
  commitTime,
  interactions
) => {
  console.log(`组件 ${id} ${phase}`);
  console.log(`实际渲染时间: ${actualDuration.toFixed(2)}ms`);
  console.log(`基准渲染时间: ${baseDuration.toFixed(2)}ms`);
};

const App = () => (
  <Profiler id="App" onRender={onRenderCallback}>
    <MainComponent />
  </Profiler>
);
```

### 2.4 VS Code 调试配置

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug iOS",
      "type": "reactnative",
      "request": "launch",
      "platform": "ios",
      "sourceMaps": true,
      "outDir": "${workspaceFolder}/.vscode/.react"
    },
    {
      "name": "Debug Android",
      "type": "reactnative",
      "request": "launch",
      "platform": "android",
      "sourceMaps": true,
      "outDir": "${workspaceFolder}/.vscode/.react"
    }
  ]
}
```

## 三、常见问题排查

```javascript
// 网络请求调试
const debugFetch = async (url, options) => {
  console.log('请求 URL:', url);
  console.log('请求选项:', JSON.stringify(options, null, 2));

  try {
    const response = await fetch(url, options);
    console.log('响应状态:', response.status);
    console.log('响应头:', Object.fromEntries(response.headers.entries()));

    const text = await response.text();
    console.log('响应体:', text);

    // 尝试解析 JSON
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  } catch (error) {
    console.error('网络错误:', error);
    throw error;
  }
};

// 布局调试
const debugLayout = () => {
  if (__DEV__) {
    // 显示布局边界
    require('react-native').setDebugLayout(true);
  }
};
```

## 四、注意事项与常见陷阱

1. **远程调试与真实环境差异**：远程调试时 JS 运行在 Chrome V8 中，性能特征与真实环境（JSC/Hermes）不同
2. **console.log 性能影响**：生产环境应移除或禁用 console.log
3. **Hermes 引擎调试**：Hermes 有独立的调试协议，不完全兼容 Chrome DevTools
4. **原生崩溃调试**：JS 调试工具无法捕获原生层崩溃，需要使用 Xcode/Android Studio
5. **网络代理调试**：使用 Charles/Proxyman 抓包时需注意 HTTPS 证书配置
