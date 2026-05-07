# Flutter 调试与性能

## 一、概念说明

Flutter 提供了丰富的调试工具，包括 DevTools、日志系统、布局调试器和性能分析器。

```dart
// 调试工具概览
/*
1. Flutter DevTools - 浏览器端调试工具
2. debugPrint - 改进的日志输出
3. Widget Inspector - Widget 树可视化
4. Performance Overlay - 性能叠加层
5. 断点调试 - IDE 集成
*/
```

## 二、调试方法

### 2.1 日志与断点

```dart
// 调试日志
void debugExample() {
  debugPrint('调试信息: ${largeData.toString()}');

  // 条件断点
  assert(() {
    if (condition) {
      debugger(); // 需要 import 'dart:developer';
    }
    return true;
  }());

  // 性能测量
  final stopwatch = Stopwatch()..start();
  expensiveOperation();
  stopwatch.stop();
  debugPrint('耗时: ${stopwatch.elapsedMilliseconds}ms');
}
```

### 2.2 布局调试

```dart
void main() {
  // 显示布局边界
  debugPaintSizeEnabled = true;

  // 显示基线
  debugPaintBaselinesEnabled = true;

  // 显示层变化
  debugRepaintRainbowEnabled = true;

  runApp(MyApp());
}
```

### 2.3 DevTools 使用

```bash
# 启动 DevTools
flutter pub global activate devtools
flutter pub global run devtools

# 或在运行时启动
flutter run --debug
# 然后按 'd' 打开 DevTools
```

```dart
// 将日志发送到 DevTools
import 'dart:developer' as developer;

void logEvent(String name, Map<String, dynamic> data) {
  developer.log(
    '事件: $name',
    name: 'com.myapp',
    level: 800,
    error: data,
  );
}

// 时间线标记
import 'dart:developer';

void performanceCriticalCode() {
  Timeline.startSync('expensive_operation');
  // 耗时操作
  Timeline.finishSync();
}
```

## 三、性能优化

### 3.1 Widget 重建优化

```dart
// 1. 使用 const 构造函数
class OptimizedWidget extends StatelessWidget {
  const OptimizedWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return const Column(
      children: [
        Text('静态文本'), // const
        Icon(Icons.star),  // const
      ],
    );
  }
}

// 2. 使用 ValueListenableBuilder 局部重建
class CounterPage extends StatefulWidget {
  const CounterPage({super.key});

  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  final ValueNotifier<int> _count = ValueNotifier(0);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text('标题不会重建'), // 不受 count 影响
        ValueListenableBuilder<int>(
          valueListenable: _count,
          builder: (context, value, child) {
            return Text('计数: $value'); // 只有这部分重建
          },
        ),
        ElevatedButton(
          onPressed: () => _count.value++,
          child: const Text('+1'),
        ),
      ],
    );
  }
}

// 3. 分离状态，避免大面积重建
class OptimizedList extends StatelessWidget {
  const OptimizedList({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: 100,
      itemBuilder: (context, index) {
        return _ListItem(index: index); // 每个项独立
      },
    );
  }
}

class _ListItem extends StatelessWidget {
  final int index;
  const _ListItem({required this.index});

  @override
  Widget build(BuildContext context) {
    return ListTile(title: Text('项目 $index'));
  }
}
```

### 3.2 图片优化

```dart
// 图片缓存与加载优化
class OptimizedImage extends StatelessWidget {
  final String url;

  const OptimizedImage({super.key, required this.url});

  @override
  Widget build(BuildContext context) {
    return Image.network(
      url,
      // 缓存尺寸
      cacheWidth: 300,
      cacheHeight: 300,
      // 加载占位符
      loadingBuilder: (context, child, loadingProgress) {
        if (loadingProgress == null) return child;
        return Center(
          child: CircularProgressIndicator(
            value: loadingProgress.expectedTotalBytes != null
                ? loadingProgress.cumulativeBytesLoaded /
                    loadingProgress.expectedTotalBytes!
                : null,
          ),
        );
      },
      // 错误处理
      errorBuilder: (context, error, stackTrace) {
        return const Icon(Icons.broken_image, size: 50);
      },
    );
  }
}
```

### 3.3 列表优化

```dart
// ListView 优化配置
ListView.builder(
  itemCount: items.length,
  // 固定高度项，提供 itemExtent
  itemExtent: 80.0,
  // 优化构建
  cacheExtent: 200,
  // 使用 const 构造的 item
  itemBuilder: (context, index) {
    return _buildOptimizedItem(items[index]);
  },
);
```

## 四、内存管理

```dart
// 内存泄漏防范
class _MyState extends State<MyWidget> {
  late AnimationController _controller;
  StreamSubscription? _subscription;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this);

    // 监听流
    _subscription = someStream.listen((data) {});

    // 定时器
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {});
  }

  @override
  void dispose() {
    // 必须清理所有资源
    _controller.dispose();
    _subscription?.cancel();
    _timer?.cancel();
    super.dispose();
  }
}
```

## 五、注意事项与常见陷阱

1. **Release 模式测试**：性能问题在 Release 模式下才准确，Debug 模式有 JIT 开销
2. **DevTools 分析**：使用 Memory 和 Performance 面板分析问题
3. **避免 rebuildAll**：不要在 build 方法中执行耗时操作
4. **图片资源管理**：大图片应使用 cacheWidth/cacheHeight 限制内存占用
5. **频繁 setState**：合并多个 setState 调用，避免频繁重建
