# Flutter 入门

## 一、概念说明

Flutter 是 Google 推出的开源 UI 工具包，使用 Dart 语言开发，通过自绘引擎（Skia/Impeller）直接绘制 UI，不依赖平台原生组件，实现真正的跨平台一致性。

```dart
// Flutter 核心特点
// 1. 自绘引擎 - 不依赖平台组件
// 2. 声明式 UI - 通过 Widget 树描述界面
// 3. 热重载 - 秒级开发体验
// 4. 高性能 - 编译为原生代码
// 5. 跨平台 - iOS、Android、Web、桌面
```

## 二、环境搭建

```bash
# 下载 Flutter SDK
# https://flutter.dev/docs/get-started/install

# 配置环境变量
export PATH="$PATH:`pwd`/flutter/bin"

# 验证安装
flutter doctor

# 创建项目
flutter create my_app
cd my_app

# 运行项目
flutter run

# 指定设备运行
flutter run -d chrome      # Web
flutter run -d macos       # macOS
flutter run -d <device_id> # 指定设备
```

```dart
// 项目结构
/*
my_app/
├── lib/
│   ├── main.dart          // 应用入口
│   ├── screens/           // 页面
│   ├── widgets/           // 组件
│   ├── models/            // 数据模型
│   ├── services/          // 服务层
│   ├── utils/             // 工具函数
│   └── theme/             // 主题配置
├── test/                  // 测试文件
├── android/               // Android 配置
├── ios/                   // iOS 配置
├── web/                   // Web 配置
├── pubspec.yaml           // 依赖配置
└── analysis_options.yaml  // 代码分析配置
*/
```

## 三、Hello World

```dart
// lib/main.dart
import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '我的第一个 Flutter 应用',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('首页'),
      ),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '你好，Flutter！',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            Text(
              '欢迎来到 Flutter 世界',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('按钮被点击了！')),
          );
        },
        child: const Icon(Icons.add),
      ),
    );
  }
}
```

## 四、Widget 基础

```dart
// StatelessWidget - 无状态组件
class GreetingText extends StatelessWidget {
  final String name;

  const GreetingText({super.key, required this.name});

  @override
  Widget build(BuildContext context) {
    return Text('你好, $name!');
  }
}

// StatefulWidget - 有状态组件
class Counter extends StatefulWidget {
  const Counter({super.key});

  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text('计数: $_count', style: const TextStyle(fontSize: 24)),
        const SizedBox(height: 16),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('+1'),
        ),
      ],
    );
  }
}
```

## 五、常用 Widget

```dart
// 常用布局 Widget 汇总
class WidgetShowcase extends StatelessWidget {
  const WidgetShowcase({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 文本
          const Text('标题', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          const Text('正文内容', style: TextStyle(fontSize: 16)),

          const SizedBox(height: 16),

          // 按钮
          ElevatedButton(onPressed: () {}, child: const Text('Elevated')),
          OutlinedButton(onPressed: () {}, child: const Text('Outlined')),
          TextButton(onPressed: () {}, child: const Text('Text')),

          const SizedBox(height: 16),

          // 图片
          Image.network('https://picsum.photos/200', width: 200, height: 200),

          const SizedBox(height: 16),

          // 图标
          const Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Icon(Icons.home, size: 32),
              Icon(Icons.search, size: 32),
              Icon(Icons.person, size: 32),
            ],
          ),

          const SizedBox(height: 16),

          // 卡片
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  const Text('卡片标题', style: TextStyle(fontSize: 18)),
                  const SizedBox(height: 8),
                  const Text('卡片内容描述'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

## 六、注意事项与常见陷阱

1. **Dart 语言基础**：学习 Flutter 前需要掌握 Dart 语言的基本语法
2. **Widget 不可变**：Widget 是不可变的，状态变化通过创建新 Widget 实现
3. **setState 使用**：只在 StatefulWidget 中使用 setState，不要在 build 中调用
4. **const 构造函数**：尽可能使用 const 构造函数，减少重建开销
5. **热重载限制**：修改 main 函数、全局变量、枚举类型需要完全重启
