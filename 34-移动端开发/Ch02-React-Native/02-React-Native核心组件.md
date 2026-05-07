# React Native 核心组件

## 一、概念说明

React Native 提供了一系列内置的核心组件，它们会自动映射到对应平台的原生 UI 元素。掌握这些组件是开发 React Native 应用的基础。

```javascript
// 核心组件概览
import {
  View,           // 容器组件，类似 HTML 的 div
  Text,           // 文本显示
  Image,          // 图片显示
  TextInput,      // 文本输入
  ScrollView,     // 可滚动容器
  FlatList,       // 高性能列表
  SectionList,    // 分组列表
  TouchableOpacity,// 可点击组件
  Pressable,      // 新版可点击组件
  Modal,          // 模态框
  StatusBar,      // 状态栏控制
  ActivityIndicator,// 加载指示器
  Switch,         // 开关
  RefreshControl, // 下拉刷新
} from 'react-native';
```

## 二、核心组件详解

### 2.1 View 和 Text

```javascript
// View 是最基本的容器组件
const ViewDemo = () => (
  <View style={{
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    padding: 16,
  }}>
    {/* 嵌套 View 构建复杂布局 */}
    <View style={{ width: 100, height: 100, backgroundColor: '#e74c3c', borderRadius: 8 }}>
      <Text style={{ color: '#fff', textAlign: 'center', marginTop: 40 }}>红色</Text>
    </View>
    <View style={{ width: 100, height: 100, backgroundColor: '#3498db', borderRadius: 8 }}>
      <Text style={{ color: '#fff', textAlign: 'center', marginTop: 40 }}>蓝色</Text>
    </View>
  </View>
);

// Text 组件支持嵌套和样式继承
const TextDemo = () => (
  <Text style={{ fontSize: 16, color: '#333' }}>
    普通文本
    <Text style={{ fontWeight: 'bold', color: '#e74c3c' }}>
      粗体红色文本
    </Text>
    <Text style={{ fontStyle: 'italic', color: '#3498db' }}>
      斜体蓝色文本
    </Text>
    {'\n'}
    <Text numberOfLines={2} ellipsizeMode="tail">
      长文本自动截断显示省略号...
    </Text>
  </Text>
);
```

### 2.2 Image 组件

```javascript
// Image 组件的各种用法
const ImageDemo = () => (
  <View>
    {/* 网络图片 */}
    <Image
      source={{ uri: 'https://example.com/photo.jpg' }}
      style={{ width: 200, height: 200 }}
      resizeMode="cover"
    />

    {/* 本地图片 */}
    <Image
      source={require('../assets/logo.png')}
      style={{ width: 100, height: 100 }}
    />

    {/* 圆形头像 */}
    <Image
      source={{ uri: 'https://example.com/avatar.jpg' }}
      style={{
        width: 80,
        height: 80,
        borderRadius: 40,
        borderWidth: 2,
        borderColor: '#fff',
      }}
    />

    {/* 背景图片 */}
    <ImageBackground
      source={{ uri: 'https://example.com/bg.jpg' }}
      style={{ width: '100%', height: 200 }}
    >
      <Text style={{ color: '#fff', fontSize: 24 }}>覆盖在图片上的文字</Text>
    </ImageBackground>
  </View>
);
```

### 2.3 TextInput 详解

```javascript
// TextInput 完整示例
const TextInputDemo = () => {
  const [value, setValue] = useState('');
  const inputRef = useRef(null);

  return (
    <View>
      {/* 基础输入框 */}
      <TextInput
        ref={inputRef}
        style={styles.input}
        placeholder="请输入用户名"
        value={value}
        onChangeText={setValue}
        // 键盘配置
        keyboardType="default"
        autoCapitalize="none"
        autoCorrect={false}
        // 安全输入（密码）
        secureTextEntry={false}
        // 返回键配置
        returnKeyType="done"
        onSubmitEditing={() => console.log('提交:', value)}
        // 字符限制
        maxLength={50}
        // 多行输入
        multiline={false}
        numberOfLines={1}
      />

      {/* 多行文本输入 */}
      <TextInput
        style={[styles.input, { height: 100, textAlignVertical: 'top' }]}
        placeholder="请输入详细描述..."
        multiline={true}
        numberOfLines={4}
        maxLength={500}
      />
    </View>
  );
};
```

### 2.4 Modal 组件

```javascript
// Modal 模态框
const ModalDemo = () => {
  const [visible, setVisible] = useState(false);

  return (
    <View>
      <TouchableOpacity onPress={() => setVisible(true)}>
        <Text>打开模态框</Text>
      </TouchableOpacity>

      <Modal
        visible={visible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>提示</Text>
            <Text style={styles.modalMessage}>确定要执行此操作吗？</Text>
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.cancelButton]}
                onPress={() => setVisible(false)}
              >
                <Text>取消</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.confirmButton]}
                onPress={() => {
                  setVisible(false);
                  // 执行操作
                }}
              >
                <Text style={{ color: '#fff' }}>确定</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};
```

### 2.5 ActivityIndicator 和 Switch

```javascript
// 加载指示器
const LoadingDemo = () => (
  <View style={{ alignItems: 'center', padding: 20 }}>
    <ActivityIndicator size="large" color="#3498db" />
    <Text style={{ marginTop: 10 }}>加载中...</Text>
  </View>
);

// 开关组件
const SwitchDemo = () => {
  const [isEnabled, setIsEnabled] = useState(false);

  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', padding: 16 }}>
      <Text style={{ flex: 1 }}>深色模式</Text>
      <Switch
        value={isEnabled}
        onValueChange={setIsEnabled}
        trackColor={{ false: '#ddd', true: '#3498db' }}
        thumbColor={isEnabled ? '#fff' : '#f4f3f4'}
      />
    </View>
  );
};
```

## 三、组件组合模式

```javascript
// 组合构建复杂界面
const ComplexScreen = () => (
  <SafeAreaView style={{ flex: 1 }}>
    {/* 顶部导航 */}
    <View style={styles.header}>
      <TouchableOpacity><Text>返回</Text></TouchableOpacity>
      <Text style={styles.headerTitle}>详情页</Text>
      <TouchableOpacity><Text>更多</Text></TouchableOpacity>
    </View>

    {/* 内容区域 */}
    <ScrollView>
      <Image source={{ uri: '...' }} style={styles.banner} />
      <View style={styles.content}>
        <Text style={styles.title}>标题</Text>
        <Text style={styles.body}>正文内容...</Text>
      </View>
    </ScrollView>

    {/* 底部操作栏 */}
    <View style={styles.footer}>
      <TouchableOpacity style={styles.primaryButton}>
        <Text style={{ color: '#fff' }}>立即购买</Text>
      </TouchableOpacity>
    </View>
  </SafeAreaView>
);
```

## 四、注意事项与常见陷阱

1. **Text 必须包裹文字**：在 React Native 中，文字必须放在 Text 组件内，不能直接放在 View 中
2. **Image 需要指定尺寸**：Image 组件必须指定 width 和 height，否则不会显示
3. **ScrollView vs FlatList**：长列表应使用 FlatList 而非 ScrollView，以获得更好的性能
4. **TextInput 键盘遮挡**：需要配合 KeyboardAvoidingView 使用，避免键盘遮挡输入框
5. **触摸反馈**：可点击组件应使用 TouchableOpacity 或 Pressable，提供视觉反馈
