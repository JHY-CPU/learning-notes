# Bloc 模式

## 一、概念说明

Bloc（Business Logic Component）是 Flutter 中流行的状态管理方案，通过事件（Event）驱动状态（State）变化，实现业务逻辑与 UI 的完全分离。

```dart
// Bloc 数据流
/*
Event (事件) -> Bloc (处理) -> State (状态) -> UI (渲染)
用户操作 -> dispatch Event -> Bloc 处理 -> emit State -> UI 重建
*/
```

## 二、Bloc 基础

```dart
// counter_event.dart
abstract class CounterEvent {}
class Increment extends CounterEvent {}
class Decrement extends CounterEvent {}
class Reset extends CounterEvent {}

// counter_state.dart
class CounterState {
  final int value;
  const CounterState(this.value);
}

// counter_bloc.dart
import 'package:flutter_bloc/flutter_bloc.dart';

class CounterBloc extends Bloc<CounterEvent, CounterState> {
  CounterBloc() : super(const CounterState(0)) {
    on<Increment>((event, emit) {
      emit(CounterState(state.value + 1));
    });
    on<Decrement>((event, emit) {
      emit(CounterState(state.value - 1));
    });
    on<Reset>((event, emit) {
      emit(const CounterState(0));
    });
  }
}
```

```dart
// UI 使用
class CounterPage extends StatelessWidget {
  const CounterPage({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (_) => CounterBloc(),
      child: Scaffold(
        body: BlocBuilder<CounterBloc, CounterState>(
          builder: (context, state) {
            return Center(child: Text('计数: ${state.value}'));
          },
        ),
        floatingActionButton: Column(
          children: [
            FloatingActionButton(
              onPressed: () => context.read<CounterBloc>().add(Increment()),
              child: const Icon(Icons.add),
            ),
            FloatingActionButton(
              onPressed: () => context.read<CounterBloc>().add(Decrement()),
              child: const Icon(Icons.remove),
            ),
          ],
        ),
      ),
    );
  }
}
```

## 三、异步 Bloc

```dart
// user_event.dart
abstract class UserEvent {}
class FetchUser extends UserEvent {
  final int id;
  FetchUser(this.id);
}
class UpdateUser extends UserEvent {
  final Map<String, dynamic> data;
  UpdateUser(this.data);
}

// user_state.dart
abstract class UserState {}
class UserInitial extends UserState {}
class UserLoading extends UserState {}
class UserLoaded extends UserState {
  final User user;
  UserLoaded(this.user);
}
class UserError extends UserState {
  final String message;
  UserError(this.message);
}

// user_bloc.dart
class UserBloc extends Bloc<UserEvent, UserState> {
  final UserRepository repository;

  UserBloc(this.repository) : super(UserInitial()) {
    on<FetchUser>((event, emit) async {
      emit(UserLoading());
      try {
        final user = await repository.getUser(event.id);
        emit(UserLoaded(user));
      } catch (e) {
        emit(UserError(e.toString()));
      }
    });
  }
}

// UI 使用
BlocProvider(
  create: (_) => UserBloc(UserRepository())..add(FetchUser(1)),
  child: BlocBuilder<UserBloc, UserState>(
    builder: (context, state) {
      if (state is UserLoading) return const CircularProgressIndicator();
      if (state is UserLoaded) return Text(state.user.name);
      if (state is UserError) return Text('错误: ${state.message}');
      return const SizedBox();
    },
  ),
);
```

## 四、Bloc 与 Cubit

```dart
// Cubit 是 Bloc 的简化版，不需要 Event
class CounterCubit extends Cubit<int> {
  CounterCubit() : super(0);

  void increment() => emit(state + 1);
  void decrement() => emit(state - 1);
}

// 使用
BlocProvider(
  create: (_) => CounterCubit(),
  child: BlocBuilder<CounterCubit, int>(
    builder: (context, count) => Text('$count'),
  ),
);
```

## 五、注意事项

1. **事件粒度**：事件应描述"用户做了什么"，而非"要变成什么状态"
2. **状态不可变**：State 类使用 final 字段，每次创建新实例
3. **Bloc 范围**：在合适的层级创建 BlocProvider，避免过深或过浅
4. **监听器**：使用 BlocListener 处理副作用（导航、弹窗）
5. **测试友好**：Bloc 天然适合单元测试，业务逻辑与 UI 分离
