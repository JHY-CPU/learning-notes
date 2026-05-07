# Flutter 测试

## 一、概念说明

Flutter 提供了完整的测试框架，包括单元测试、Widget 测试和集成测试三个层次。

```dart
// 测试金字塔
/*
      / E2E \       <- integration_test/
     / Widget \     <- test/ widget tests
    /  Unit    \    <- test/ unit tests
*/

// pubspec.yaml
dev_dependencies:
  flutter_test:
    sdk: flutter
  integration_test:
    sdk: flutter
  mockito: ^5.4.0
  bloc_test: ^9.1.0
```

## 二、单元测试

```dart
// models/counter.dart
class Counter {
  int _value = 0;
  int get value => _value;
  void increment() => _value++;
  void decrement() {
    if (_value > 0) _value--;
  }
  void reset() => _value = 0;
}

// test/models/counter_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:my_app/models/counter.dart';

void main() {
  group('Counter', () {
    late Counter counter;

    setUp(() {
      counter = Counter();
    });

    test('初始值应该为0', () {
      expect(counter.value, 0);
    });

    test('increment 应该增加1', () {
      counter.increment();
      expect(counter.value, 1);
    });

    test('decrement 应该减少1', () {
      counter.increment();
      counter.increment();
      counter.decrement();
      expect(counter.value, 1);
    });

    test('decrement 不应该小于0', () {
      counter.decrement();
      expect(counter.value, 0);
    });

    test('reset 应该重置为0', () {
      counter.increment();
      counter.increment();
      counter.reset();
      expect(counter.value, 0);
    });
  });
}
```

### 2.1 异步测试

```dart
// services/user_service.dart
class UserService {
  final ApiClient _api;
  UserService(this._api);

  Future<User> getUser(int id) async {
    final data = await _api.get('/users/$id');
    return User.fromJson(data);
  }
}

// test/services/user_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';
import 'user_service_test.mocks.dart';

@GenerateMocks([ApiClient])
void main() {
  late MockApiClient mockApi;
  late UserService service;

  setUp(() {
    mockApi = MockApiClient();
    service = UserService(mockApi);
  });

  test('getUser 应该返回用户数据', () async {
    when(mockApi.get('/users/1')).thenAnswer(
      (_) async => {'id': 1, 'name': '张三'},
    );

    final user = await service.getUser(1);

    expect(user.name, '张三');
    verify(mockApi.get('/users/1')).called(1);
  });

  test('getUser 网络错误应该抛出异常', () {
    when(mockApi.get('/users/1')).thenThrow(
      Exception('网络错误'),
    );

    expect(() => service.getUser(1), throwsException);
  });
}
```

## 三、Widget 测试

```dart
// widgets/counter_widget.dart
class CounterWidget extends StatefulWidget {
  const CounterWidget({super.key});

  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _count = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('计数: $_count', key: const Key('counter_text')),
        ElevatedButton(
          key: const Key('increment_button'),
          onPressed: () => setState(() => _count++),
          child: const Text('+1'),
        ),
      ],
    );
  }
}

// test/widgets/counter_widget_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:my_app/widgets/counter_widget.dart';

void main() {
  testWidgets('应该显示初始计数为0', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: CounterWidget()));

    expect(find.text('计数: 0'), findsOneWidget);
    expect(find.byKey(const Key('increment_button')), findsOneWidget);
  });

  testWidgets('点击按钮应该增加计数', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: CounterWidget()));

    await tester.tap(find.byKey(const Key('increment_button')));
    await tester.pump(); // 触发重建

    expect(find.text('计数: 1'), findsOneWidget);
    expect(find.text('计数: 0'), findsNothing);
  });

  testWidgets('多次点击应该正确计数', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: CounterWidget()));

    for (int i = 0; i < 5; i++) {
      await tester.tap(find.byType(ElevatedButton));
      await tester.pump();
    }

    expect(find.text('计数: 5'), findsOneWidget);
  });
}
```

### 3.1 滚动测试

```dart
testWidgets('列表应该可以滚动', (tester) async {
  await tester.pumpWidget(MaterialApp(
    home: ListView.builder(
      itemCount: 100,
      itemBuilder: (_, i) => ListTile(title: Text('项目 $i')),
    ),
  ));

  // 初始可见项
  expect(find.text('项目 0'), findsOneWidget);
  expect(find.text('项目 50'), findsNothing);

  // 滚动到底部
  await tester.fling(find.byType(ListView), const Offset(0, -500), 1000);
  await tester.pumpAndSettle();

  expect(find.text('项目 50'), findsOneWidget);
});
```

## 四、集成测试

```dart
// integration_test/app_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:my_app/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('完整登录流程', (tester) async {
    app.main();
    await tester.pumpAndSettle();

    // 找到输入框并输入
    await tester.enterText(find.byKey(const Key('email_input')), 'test@example.com');
    await tester.enterText(find.byKey(const Key('password_input')), 'password123');

    // 点击登录按钮
    await tester.tap(find.byKey(const Key('login_button')));
    await tester.pumpAndSettle();

    // 验证跳转到首页
    expect(find.text('欢迎回来'), findsOneWidget);
    expect(find.byType(app.HomePage), findsOneWidget);
  });
}
```

```bash
# 运行测试
flutter test                        # 单元和 Widget 测试
flutter test test/models/           # 特定目录
flutter test --coverage             # 覆盖率
flutter test integration_test/      # 集成测试
flutter drive --driver=test_driver/integration_test.dart \
  --target=integration_test/app_test.dart
```

## 五、注意事项与常见陷阱

1. **pumpAndSettle vs pump**：有动画的测试使用 `pumpAndSettle` 等待动画完成
2. **Mock 的使用**：只 Mock 外部依赖，不要 Mock 被测试对象
3. **测试隔离**：每个测试应独立运行，使用 setUp 创建测试数据
4. **Key 的使用**：Widget 测试中使用 Key 精确定位元素
5. **集成测试环境**：集成测试需要真机或模拟器运行，CI 环境需要配置
