# React Native 入门

## 一、概念说明

React Native 是 Facebook 于 2015 年推出的跨平台移动应用开发框架，使用 JavaScript 和 React 语法构建原生移动应用。它通过 JavaScript Bridge 将 React 组件映射为原生 UI 组件。

```javascript
// React Native 基础项目结构
/*
MyApp/
├── src/
│   ├── screens/        // 页面组件
│   ├── components/     // 通用组件
│   ├── navigation/     // 导航配置
│   ├── services/       // API 服务
│   ├── store/          // 状态管理
│   └── utils/          // 工具函数
├── android/            // Android 原生代码
├── ios/                // iOS 原生代码
├── App.tsx             // 应用入口
├── index.js            // 注册入口
├── package.json        // 依赖配置
└── babel.config.js     // Babel 配置
*/
```

## 二、环境搭建

### 2.1 开发环境配置

```bash
# 安装 React Native CLI
npm install -g react-native-cli

# 创建新项目
npx react-native init MyApp --template react-native-template-typescript

# 运行项目
cd MyApp
npx react-native run-android   # Android
npx react-native run-ios       # iOS (需要 macOS)

# 启动 Metro 服务器
npx react-native start
```

### 2.2 项目配置

```javascript
// babel.config.js
module.exports = {
  presets: ['module:metro-react-native-babel-preset'],
  plugins: [
    ['module-resolver', {
      root: ['./src'],
      alias: {
        '@': './src',
        '@components': './src/components',
        '@screens': './src/screens',
      },
    }],
  ],
};
```

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "esnext",
    "module": "commonjs",
    "lib": ["es2019"],
    "allowJs": true,
    "jsx": "react-native",
    "strict": true,
    "moduleResolution": "node",
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    },
    "esModuleInterop": true,
    "skipLibCheck": true
  }
}
```

## 三、基础组件

### 3.1 核心组件

```javascript
import React from 'react';
import {
  View,
  Text,
  Image,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  StatusBar,
} from 'react-native';

const BasicComponents = () => {
  const [inputValue, setInputValue] = React.useState('');

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* 文本组件 */}
        <Text style={styles.title}>React Native 基础组件</Text>
        <Text style={styles.subtitle}>跨平台移动开发框架</Text>

        {/* 图片组件 */}
        <Image
          source={{ uri: 'https://reactnative.dev/img/tiny_logo.png' }}
          style={styles.image}
          resizeMode="contain"
        />

        {/* 输入框 */}
        <TextInput
          style={styles.input}
          placeholder="请输入内容..."
          value={inputValue}
          onChangeText={setInputValue}
        />

        {/* 按钮 */}
        <TouchableOpacity
          style={styles.button}
          onPress={() => console.log('输入内容:', inputValue)}
        >
          <Text style={styles.buttonText}>提交</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5' },
  scrollContent: { padding: 16 },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 8 },
  subtitle: { fontSize: 16, color: '#666', marginBottom: 16 },
  image: { width: 100, height: 100, alignSelf: 'center', marginBottom: 16 },
  input: {
    borderWidth: 1, borderColor: '#ddd', borderRadius: 8,
    padding: 12, marginBottom: 16, backgroundColor: '#fff',
  },
  button: {
    backgroundColor: '#3498db', padding: 14, borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});

export default BasicComponents;
```

### 3.2 Flexbox 布局

```javascript
// React Native 布局系统
const FlexLayoutDemo = () => (
  <View style={styles.container}>
    {/* 水平排列 */}
    <View style={styles.row}>
      <View style={[styles.box, { backgroundColor: '#e74c3c' }]} />
      <View style={[styles.box, { backgroundColor: '#3498db' }]} />
      <View style={[styles.box, { backgroundColor: '#2ecc71' }]} />
    </View>

    {/* 垂直居中 */}
    <View style={styles.centered}>
      <Text>居中内容</Text>
    </View>

    {/* 等分排列 */}
    <View style={styles.row}>
      <View style={[styles.flexBox, { backgroundColor: '#9b59b6' }]} />
      <View style={[styles.flexBox, { backgroundColor: '#f39c12' }]} />
    </View>
  </View>
);

const layoutStyles = StyleSheet.create({
  row: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 16 },
  box: { width: 80, height: 80, borderRadius: 8 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  flexBox: { flex: 1, height: 60, marginHorizontal: 4, borderRadius: 8 },
});
```

## 四、注意事项与常见陷阱

1. **平台差异**：某些样式在 iOS 和 Android 上表现不同，如阴影效果需使用不同 API
2. **JS Bridge 性能**：频繁的跨线程通信会影响性能，尽量使用原生驱动动画
3. **热重载限制**：热重载对某些原生模块修改无效，需要完全重新构建
4. **内存泄漏**：使用定时器和事件监听时，记得在组件卸载时清理
5. **版本兼容性**：React Native 版本升级可能带来破坏性变更，需要仔细阅读迁移指南
