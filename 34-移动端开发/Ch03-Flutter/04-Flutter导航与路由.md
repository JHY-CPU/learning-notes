# Flutter 导航与路由

## 一、概念说明

Flutter 的导航系统基于 Navigator 和 Route，支持声明式和命令式两种导航方式。从 Flutter 2.0 开始，推荐使用声明式路由（GoRouter）。

```dart
// 导航系统概览
/*
1. 命名路由 - Navigator.pushNamed()
2. 匿名路由 - Navigator.push(MaterialPageRoute())
3. 声明式路由 - GoRouter (推荐)
*/
```

## 二、基础导航

### 2.1 匿名路由

```dart
// 页面跳转
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => const DetailPage(id: 123),
              ),
            );
          },
          child: const Text('跳转到详情页'),
        ),
      ),
    );
  }
}

// 接收参数
class DetailPage extends StatelessWidget {
  final int id;

  const DetailPage({super.key, required this.id});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('详情')),
      body: Center(child: Text('项目ID: $id')),
    );
  }
}
```

### 2.2 命名路由

```dart
// 注册路由表
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/',
      routes: {
        '/': (context) => const HomePage(),
        '/detail': (context) => const DetailPage(id: 0),
        '/settings': (context) => const SettingsPage(),
      },
      // 带参数的路由
      onGenerateRoute: (settings) {
        if (settings.name == '/detail') {
          final args = settings.arguments as Map<String, dynamic>;
          return MaterialPageRoute(
            builder: (context) => DetailPage(id: args['id']),
          );
        }
        return null;
      },
    );
  }
}

// 使用命名路由跳转
ElevatedButton(
  onPressed: () {
    Navigator.pushNamed(context, '/detail', arguments: {'id': 456});
  },
  child: const Text('命名路由跳转'),
);
```

### 2.3 导航操作

```dart
// 各种导航操作
class NavigationDemo {
  // 普通跳转（可返回）
  static void push(BuildContext context, Widget page) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => page));
  }

  // 替换当前页面（返回到上一个页面）
  static void pushReplacement(BuildContext context, Widget page) {
    Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => page));
  }

  // 清空栈并跳转（登录后跳首页）
  static void pushAndClear(BuildContext context, Widget page) {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => page),
      (route) => false, // 移除所有路由
    );
  }

  // 返回
  static void pop(BuildContext context) {
    Navigator.pop(context);
  }

  // 带返回值的返回
  static void popWithResult(BuildContext context, dynamic result) {
    Navigator.pop(context, result);
  }

  // 返回到指定页面
  static void popUntil(BuildContext context, String routeName) {
    Navigator.popUntil(context, ModalRoute.withName(routeName));
  }
}
```

## 三、GoRouter（推荐）

```dart
import 'package:go_router/go_router.dart';

// 路由配置
final router = GoRouter(
  initialLocation: '/',
  redirect: (context, state) {
    // 认证检查
    final isLoggedIn = AuthService.instance.isLoggedIn;
    if (!isLoggedIn && state.matchedLocation.startsWith('/protected')) {
      return '/login';
    }
    return null;
  },
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const HomePage(),
    ),
    GoRoute(
      path: '/detail/:id',
      builder: (context, state) {
        final id = int.parse(state.pathParameters['id']!);
        return DetailPage(id: id);
      },
    ),
    GoRoute(
      path: '/login',
      builder: (context, state) => const LoginPage(),
    ),
    // 嵌套路由
    ShellRoute(
      builder: (context, state, child) {
        return MainShell(child: child);
      },
      routes: [
        GoRoute(
          path: '/home',
          builder: (context, state) => const HomeTab(),
        ),
        GoRoute(
          path: '/profile',
          builder: (context, state) => const ProfileTab(),
        ),
      ],
    ),
  ],
);

// 使用 GoRouter
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      routerConfig: router,
    );
  }
}

// 跳转
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ElevatedButton(
          onPressed: () => context.go('/detail/123'),
          child: const Text('跳转 (替换当前)'),
        ),
        ElevatedButton(
          onPressed: () => context.push('/detail/456'),
          child: const Text('跳转 (压入栈)'),
        ),
        ElevatedButton(
          onPressed: () => context.go('/home'),
          child: const Text('返回首页'),
        ),
      ],
    );
  }
}
```

## 四、页面过渡动画

```dart
// 自定义过渡动画
class FadeRoute extends PageRouteBuilder {
  final Widget page;

  FadeRoute({required this.page})
      : super(
          pageBuilder: (context, animation, secondaryAnimation) => page,
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(opacity: animation, child: child);
          },
          transitionDuration: const Duration(milliseconds: 300),
        );
}

// 滑动过渡
class SlideRoute extends PageRouteBuilder {
  final Widget page;

  SlideRoute({required this.page})
      : super(
          pageBuilder: (context, animation, secondaryAnimation) => page,
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            final tween = Tween(begin: const Offset(1.0, 0.0), end: Offset.zero)
                .chain(CurveTween(curve: Curves.easeInOut));
            return SlideTransition(position: animation.drive(tween), child: child);
          },
        );
}

// 使用
Navigator.push(context, FadeRoute(page: const DetailPage(id: 1)));
```

## 五、注意事项与常见陷阱

1. **不要在 build 中执行导航**：可能导致重复导航，应在事件回调中执行
2. **异步导航结果**：使用 `await Navigator.push()` 获取返回值
3. **路由栈管理**：合理规划路由栈深度，避免内存占用过大
4. **深层链接**：配置 GoRouter 的 `redirect` 处理认证和权限
5. **返回键处理**：使用 `WillPopScope` 拦截返回操作，防止误操作
