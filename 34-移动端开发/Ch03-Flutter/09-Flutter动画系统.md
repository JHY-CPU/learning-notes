# Flutter 动画系统

## 一、概念说明

Flutter 动画系统提供了从简单到复杂的完整动画解决方案。核心包括 AnimationController、Tween、Curve 和各种内置动画 Widget。

```dart
// 动画系统层次
/*
1. 隐式动画 - AnimatedContainer, AnimatedOpacity 等
2. 显式动画 - AnimationController + AnimatedBuilder
3. 物理动画 - SpringSimulation, GravitySimulation
*/
```

## 二、隐式动画

### 2.1 AnimatedContainer

```dart
class AnimatedContainerDemo extends StatefulWidget {
  const AnimatedContainerDemo({super.key});

  @override
  State<AnimatedContainerDemo> createState() => _AnimatedContainerDemoState();
}

class _AnimatedContainerDemoState extends State<AnimatedContainerDemo> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => setState(() => _expanded = !_expanded),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        width: _expanded ? 200 : 100,
        height: _expanded ? 200 : 100,
        decoration: BoxDecoration(
          color: _expanded ? Colors.blue : Colors.red,
          borderRadius: BorderRadius.circular(_expanded ? 20 : 50),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(_expanded ? 0.3 : 0.1),
              blurRadius: _expanded ? 20 : 5,
            ),
          ],
        ),
        child: Center(
          child: Text(
            '点击我',
            style: TextStyle(color: Colors.white, fontSize: _expanded ? 20 : 14),
          ),
        ),
      ),
    );
  }
}
```

### 2.2 其他隐式动画

```dart
class ImplicitAnimationsDemo extends StatefulWidget {
  const ImplicitAnimationsDemo({super.key});

  @override
  State<ImplicitAnimationsDemo> createState() => _ImplicitAnimationsDemoState();
}

class _ImplicitAnimationsDemoState extends State<ImplicitAnimationsDemo> {
  bool _visible = true;
  double _padding = 16;
  Alignment _alignment = Alignment.center;
  TextStyle _style = const TextStyle(fontSize: 16);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // AnimatedOpacity - 透明度动画
        AnimatedOpacity(
          opacity: _visible ? 1.0 : 0.0,
          duration: const Duration(milliseconds: 500),
          child: const Text('淡入淡出文本'),
        ),
        ElevatedButton(
          onPressed: () => setState(() => _visible = !_visible),
          child: const Text('切换可见性'),
        ),

        // AnimatedPadding - 内边距动画
        AnimatedPadding(
          padding: EdgeInsets.all(_padding),
          duration: const Duration(milliseconds: 300),
          child: Container(width: 100, height: 100, color: Colors.blue),
        ),
        Slider(
          value: _padding,
          min: 0,
          max: 50,
          onChanged: (v) => setState(() => _padding = v),
        ),

        // AnimatedAlign - 对齐动画
        SizedBox(
          width: 200,
          height: 100,
          child: AnimatedAlign(
            alignment: _alignment,
            duration: const Duration(milliseconds: 500),
            curve: Curves.bounceOut,
            child: const FlutterLogo(size: 50),
          ),
        ),
        ElevatedButton(
          onPressed: () {
            setState(() {
              _alignment = _alignment == Alignment.center
                  ? Alignment.topRight
                  : Alignment.center;
            });
          },
          child: const Text('移动'),
        ),
      ],
    );
  }
}
```

## 三、显式动画

### 3.1 AnimationController

```dart
class ExplicitAnimationDemo extends StatefulWidget {
  const ExplicitAnimationDemo({super.key});

  @override
  State<ExplicitAnimationDemo> createState() => _ExplicitAnimationDemoState();
}

class _ExplicitAnimationDemoState extends State<ExplicitAnimationDemo>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  late Animation<Color?> _colorAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );

    // Tween 动画
    _animation = Tween<double>(begin: 0, end: 2 * pi)
        .chain(CurveTween(curve: Curves.easeInOut))
        .animate(_controller);

    // 颜色动画
    _colorAnimation = ColorTween(begin: Colors.blue, end: Colors.red)
        .animate(_controller);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        AnimatedBuilder(
          animation: _animation,
          builder: (context, child) {
            return Transform.rotate(
              angle: _animation.value,
              child: child,
            );
          },
          child: Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: _colorAnimation.value,
              borderRadius: BorderRadius.circular(16),
            ),
          ),
        ),
        const SizedBox(height: 32),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _controller.forward(),
              child: const Text('正向'),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: () => _controller.reverse(),
              child: const Text('反向'),
            ),
            const SizedBox(width: 8),
            ElevatedButton(
              onPressed: () => _controller.repeat(),
              child: const Text('重复'),
            ),
          ],
        ),
      ],
    );
  }
}
```

### 3.2 TweenSequence 动画序列

```dart
class TweenSequenceDemo extends StatefulWidget {
  const TweenSequenceDemo({super.key});

  @override
  State<TweenSequenceDemo> createState() => _TweenSequenceDemoState();
}

class _TweenSequenceDemoState extends State<TweenSequenceDemo>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    );

    _animation = TweenSequence<double>([
      TweenSequenceItem(tween: Tween(begin: 0.0, end: 1.0), weight: 40),
      TweenSequenceItem(tween: ConstantTween(1.0), weight: 20),
      TweenSequenceItem(
        tween: Tween(begin: 1.0, end: 0.0).chain(CurveTween(curve: Curves.easeOut)),
        weight: 40,
      ),
    ]).animate(_controller);
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Opacity(
          opacity: _animation.value,
          child: Transform.scale(
            scale: _animation.value,
            child: child,
          ),
        );
      },
      child: Container(
        width: 100,
        height: 100,
        color: Colors.green,
        child: const Center(child: Text('序列动画')),
      ),
    );
  }
}
```

## 四、Hero 动画

```dart
// 页面 A
class ListPage extends StatelessWidget {
  const ListPage({super.key});

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 3),
      itemCount: 20,
      itemBuilder: (context, index) {
        return GestureDetector(
          onTap: () {
            Navigator.push(context, MaterialPageRoute(
              builder: (_) => DetailPage(index: index),
            ));
          },
          child: Hero(
            tag: 'image_$index',
            child: Image.network('https://picsum.photos/200?random=$index'),
          ),
        );
      },
    );
  }
}

// 页面 B
class DetailPage extends StatelessWidget {
  final int index;
  const DetailPage({super.key, required this.index});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: Center(
        child: Hero(
          tag: 'image_$index',
          child: Image.network('https://picsum.photos/400?random=$index'),
        ),
      ),
    );
  }
}
```

## 五、注意事项与常见陷阱

1. **AnimationController 必须 dispose**：在 dispose 中调用 `_controller.dispose()`
2. **vsync 使用**：需要 `TickerProviderStateMixin`，不要在 initState 之外创建
3. **性能优化**：使用 `AnimatedBuilder` 只重建动画相关的 Widget
4. **useNativeDriver**：Flutter 中动画默认在 UI 线程执行，不需要额外配置
5. **Curves 选择**：根据场景选择合适的动画曲线，避免过于花哨
