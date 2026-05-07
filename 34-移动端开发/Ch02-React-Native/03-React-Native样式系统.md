# React Native 样式系统

## 一、概念说明

React Native 使用 JavaScript 对象来定义样式，类似 CSS 但有所不同。样式系统基于 Flexbox 布局，且不支持所有 CSS 属性。

```javascript
// 样式定义的三种方式
import { StyleSheet, View, Text } from 'react-native';

// 1. StyleSheet.create（推荐）
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});

// 2. 内联样式
<View style={{ padding: 16, backgroundColor: '#f5f5f5' }}>
  <Text>内联样式</Text>
</View>

// 3. 样式数组（合并多个样式）
<View style={[styles.container, styles.extra]}>
  <Text>组合样式</Text>
</View>
```

## 二、常用样式属性

### 2.1 布局属性

```javascript
const layoutStyles = StyleSheet.create({
  // Flex 布局
  container: {
    flex: 1,                      // 占据剩余空间
    flexDirection: 'row',         // 主轴方向: row | column
    justifyContent: 'center',     // 主轴对齐: flex-start | center | flex-end | space-between | space-around
    alignItems: 'center',         // 交叉轴对齐: flex-start | center | flex-end | stretch
    flexWrap: 'nowrap',           // 换行: nowrap | wrap
  },
  // 定位
  absoluteElement: {
    position: 'absolute',
    top: 0,
    right: 0,
    width: 50,
    height: 50,
  },
  // 尺寸
  box: {
    width: '50%',                 // 百分比
    height: 200,                  // 固定值
    minWidth: 100,
    maxWidth: 400,
  },
  // 间距
  padded: {
    margin: 16,                   // 外边距
    padding: 12,                  // 内边距
    marginHorizontal: 8,          // 水平外边距
    paddingVertical: 16,          // 垂直内边距
  },
});
```

### 2.2 文本样式

```javascript
const textStyles = StyleSheet.create({
  heading: {
    fontSize: 24,
    fontWeight: 'bold',           // normal | bold | 100-900
    fontFamily: 'System',         // 系统字体
    color: '#333333',
    textAlign: 'center',          // auto | left | center | right
    lineHeight: 32,               // 行高
    letterSpacing: 1,             // 字间距
    textDecorationLine: 'none',   // none | underline | line-through
  },
  body: {
    fontSize: 16,
    fontWeight: '400',
    color: '#666666',
    lineHeight: 24,
  },
  caption: {
    fontSize: 12,
    color: '#999999',
  },
});
```

### 2.3 背景与边框

```javascript
const visualStyles = StyleSheet.create({
  card: {
    // 背景
    backgroundColor: '#ffffff',
    // 圆角
    borderRadius: 12,
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
    // 边框
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderStyle: 'solid',         // solid | dashed | dotted
  },
  // iOS 阴影
  shadowIOS: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  // Android 阴影
  shadowAndroid: {
    elevation: 4,
  },
});
```

### 2.4 平台特定样式

```javascript
import { Platform, StyleSheet } from 'react-native';

const platformStyles = StyleSheet.create({
  header: {
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
      },
      android: {
        elevation: 4,
      },
    }),
    backgroundColor: '#fff',
    height: Platform.OS === 'ios' ? 44 : 56,
  },
  // 按平台使用不同字体
  text: {
    fontFamily: Platform.select({
      ios: 'San Francisco',
      android: 'Roboto',
    }),
    fontSize: 16,
  },
});
```

## 三、样式最佳实践

### 3.1 样式组织

```javascript
// 按功能组织样式文件
// styles/colors.js
export const colors = {
  primary: '#3498db',
  primaryLight: '#5dade2',
  primaryDark: '#2980b9',
  success: '#2ecc71',
  warning: '#f39c12',
  danger: '#e74c3c',
  text: '#333333',
  textSecondary: '#666666',
  textLight: '#999999',
  background: '#f5f5f5',
  white: '#ffffff',
  border: '#e0e0e0',
};

// styles/spacing.js
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
};

// styles/typography.js
export const typography = {
  h1: { fontSize: 28, fontWeight: 'bold', lineHeight: 36 },
  h2: { fontSize: 22, fontWeight: 'bold', lineHeight: 30 },
  h3: { fontSize: 18, fontWeight: '600', lineHeight: 26 },
  body: { fontSize: 16, fontWeight: '400', lineHeight: 24 },
  caption: { fontSize: 12, fontWeight: '400', lineHeight: 18 },
};
```

### 3.2 动态样式

```javascript
// 根据状态生成动态样式
const getButtonStyle = (variant, disabled) => {
  const baseStyle = {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
  };

  const variants = {
    primary: { backgroundColor: disabled ? '#bdc3c7' : '#3498db' },
    secondary: { backgroundColor: 'transparent', borderWidth: 1, borderColor: '#3498db' },
    danger: { backgroundColor: disabled ? '#bdc3c7' : '#e74c3c' },
  };

  return StyleSheet.create({
    button: { ...baseStyle, ...variants[variant] },
  }).button;
};

// 使用示例
const Button = ({ variant = 'primary', disabled, children }) => (
  <TouchableOpacity
    style={getButtonStyle(variant, disabled)}
    disabled={disabled}
  >
    <Text style={{ color: variant === 'secondary' ? '#3498db' : '#fff' }}>
      {children}
    </Text>
  </TouchableOpacity>
);
```

## 四、注意事项与常见陷阱

1. **不支持的 CSS 属性**：不支持 `box-shadow`（需分平台处理）、`overflow: visible` 在 Android 上不支持
2. **Flexbox 差异**：默认 `flexDirection: 'column'`（不同于 Web 的 `row`），`flex` 只接受单个数字
3. **样式不继承**：除 Text 嵌套外，样式不会从父组件继承
4. **数字不需要单位**：尺寸值直接使用数字，不需要 'px' 后缀
5. **StyleSheet.create 优势**：提供样式验证和性能优化，推荐始终使用
