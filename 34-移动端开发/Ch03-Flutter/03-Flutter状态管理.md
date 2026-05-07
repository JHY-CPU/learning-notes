# Flutter 状态管理

## 一、概念说明

状态管理是 Flutter 应用架构的核心。根据状态的作用范围，可分为局部状态（Widget 内部）和全局状态（跨 Widget 共享）。

```dart
// 状态分类
/*
1. 局部状态 - setState (Ephemeral State)
2. 全局状态 - Provider / Riverpod / Bloc / GetX (App State)
*/
```

## 二、局部状态管理

### 2.1 setState

```dart
class CounterPage extends StatefulWidget {
  const CounterPage({super.key});

  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('计数: $_count', style: const TextStyle(fontSize: 32)),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  onPressed: () => setState(() => _count--),
                  icon: const Icon(Icons.remove),
                ),
                IconButton(
                  onPressed: _increment,
                  icon: const Icon(Icons.add),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
```

### 2.2 InheritedWidget

```dart
// 通过 InheritedWidget 在 Widget 树中共享数据
class ThemeInherited extends InheritedWidget {
  final Color primaryColor;
  final double fontSize;
  final VoidCallback toggleTheme;

  const ThemeInherited({
    super.key,
    required this.primaryColor,
    required this.fontSize,
    required this.toggleTheme,
    required super.child,
  });

  static ThemeInherited? of(BuildContext context) {
    return context.dependOnInheritedWidgetOfExactType<ThemeInherited>();
  }

  @override
  bool updateShouldNotify(ThemeInherited oldWidget) {
    return primaryColor != oldWidget.primaryColor ||
           fontSize != oldWidget.fontSize;
  }
}

// 使用
class ThemedText extends StatelessWidget {
  const ThemedText({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = ThemeInherited.of(context)!;
    return Text(
      '主题文本',
      style: TextStyle(
        color: theme.primaryColor,
        fontSize: theme.fontSize,
      ),
    );
  }
}
```

## 三、全局状态管理 - Provider

### 3.1 基础用法

```dart
import 'package:provider/provider.dart';

// 创建 ChangeNotifier
class CounterNotifier extends ChangeNotifier {
  int _count = 0;
  int get count => _count;

  void increment() {
    _count++;
    notifyListeners();
  }

  void decrement() {
    _count--;
    notifyListeners();
  }
}

// 注册 Provider
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => CounterNotifier()),
        ChangeNotifierProvider(create: (_) => UserNotifier()),
        Provider(create: (_) => ApiService()),
      ],
      child: MaterialApp(home: HomePage()),
    );
  }
}

// 消费 Provider
class CounterWidget extends StatelessWidget {
  const CounterWidget({super.key});

  @override
  Widget build(BuildContext context) {
    // 方式1: context.watch (会重建)
    final counter = context.watch<CounterNotifier>();

    return Column(
      children: [
        Text('计数: ${counter.count}'),
        ElevatedButton(
          onPressed: () => context.read<CounterNotifier>().increment(),
          child: const Text('+1'),
        ),
      ],
    );
  }
}

// 方式2: Consumer Widget
class CounterConsumer extends StatelessWidget {
  const CounterConsumer({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<CounterNotifier>(
      builder: (context, counter, child) {
        return Column(
          children: [
            child!, // 不依赖状态的子 Widget 不会重建
            Text('计数: ${counter.count}'),
          ],
        );
      },
      child: const Text('这个文本不会重建'),
    );
  }
}
```

### 3.2 异步状态管理

```dart
class UserNotifier extends ChangeNotifier {
  User? _user;
  bool _loading = false;
  String? _error;

  User? get user => _user;
  bool get loading => _loading;
  String? get error => _error;

  Future<void> fetchUser(int id) async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      _user = await ApiService().getUser(id);
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }
}

// 使用
class UserPage extends StatefulWidget {
  const UserPage({super.key});

  @override
  State<UserPage> createState() => _UserPageState();
}

class _UserPageState extends State<UserPage> {
  @override
  void initState() {
    super.initState();
    context.read<UserNotifier>().fetchUser(1);
  }

  @override
  Widget build(BuildContext context) {
    final userState = context.watch<UserNotifier>();

    if (userState.loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (userState.error != null) {
      return Center(child: Text('错误: ${userState.error}'));
    }

    return Text('用户: ${userState.user?.name}');
  }
}
```

## 四、Riverpod（Provider 的升级版）

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

// 定义 Provider
final counterProvider = StateNotifierProvider<CounterNotifier, int>((ref) {
  return CounterNotifier();
});

final userProvider = FutureProvider.family<User, int>((ref, userId) async {
  return ApiService().getUser(userId);
});

// 使用
class CounterPage extends ConsumerWidget {
  const CounterPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final count = ref.watch(counterProvider);

    return Column(
      children: [
        Text('计数: $count'),
        ElevatedButton(
          onPressed: () => ref.read(counterProvider.notifier).increment(),
          child: const Text('+1'),
        ),
      ],
    );
  }
}
```

## 五、状态管理方案对比

| 方案 | 复杂度 | 学习曲线 | 适用场景 |
|------|--------|----------|----------|
| setState | 低 | 低 | 局部状态 |
| InheritedWidget | 中 | 中 | 简单共享 |
| Provider | 中 | 中 | 中小型应用 |
| Riverpod | 中高 | 中高 | 中大型应用 |
| Bloc | 高 | 高 | 复杂业务逻辑 |
| GetX | 低 | 低 | 快速开发 |

## 六、注意事项与常见陷阱

1. **不要在 build 中创建 ChangeNotifier**：应在外部创建或使用 `create` 参数
2. **合理拆分状态**：按业务功能拆分，避免单一巨大的状态类
3. **使用 Selector 优化**：只订阅需要的部分状态，减少重建
4. **异步状态处理**：妥善处理 loading、error、data 三种状态
5. **dispose 清理**：在状态类的 dispose 中清理资源（定时器、监听器等）
