# Flutter 布局系统

## 一、概念说明

Flutter 的布局系统基于 Widget 嵌套，通过约束（Constraints）向下传递，尺寸（Size）向上传递的机制实现灵活的布局。

```dart
// 布局核心原则
// 1. 约束向下传递 - 父 Widget 告诉子 Widget 可用空间
// 2. 尺寸向上传递 - 子 Widget 告诉父 Widget 自己的大小
// 3. 位置由父决定 - 父 Widget 决定子 Widget 在自身中的位置

// 布局 Widget 分类
/*
单子布局: Container, Padding, Center, Align, SizedBox, ...
多子布局: Row, Column, Stack, Wrap, Flex, ListView, ...
*/
```

## 二、单子布局 Widget

### 2.1 Container

```dart
// Container - 最常用的布局容器
class ContainerDemo extends StatelessWidget {
  const ContainerDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      // 尺寸
      width: 200,
      height: 200,
      // 约束
      constraints: const BoxConstraints(
        minWidth: 100,
        maxWidth: 300,
        minHeight: 100,
        maxHeight: 300,
      ),
      // 内边距
      padding: const EdgeInsets.all(16),
      // 外边距
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      // 装饰
      decoration: BoxDecoration(
        color: Colors.blue,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.2),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
        gradient: const LinearGradient(
          colors: [Colors.blue, Colors.purple],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      // 子 Widget
      child: const Text(
        'Container',
        style: TextStyle(color: Colors.white, fontSize: 18),
      ),
    );
  }
}
```

### 2.2 Padding、SizedBox、Expanded

```dart
class SingleChildLayoutDemo extends StatelessWidget {
  const SingleChildLayoutDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Padding - 添加内边距
        Padding(
          padding: const EdgeInsets.all(16),
          child: const Text('带内边距的文本'),
        ),

        // SizedBox - 指定尺寸/间距
        const SizedBox(height: 16),   // 垂直间距
        SizedBox(
          width: double.infinity,     // 充满宽度
          height: 48,
          child: ElevatedButton(
            onPressed: () {},
            child: const Text('全宽按钮'),
          ),
        ),

        // ConstrainedBox - 约束
        ConstrainedBox(
          constraints: const BoxConstraints(
            minWidth: 100,
            maxWidth: 200,
            minHeight: 50,
          ),
          child: const Text('有约束的文本'),
        ),

        // AspectRatio - 宽高比
        AspectRatio(
          aspectRatio: 16 / 9,
          child: Container(
            color: Colors.blue,
            child: const Center(child: Text('16:9 比例')),
          ),
        ),
      ],
    );
  }
}
```

## 三、多子布局 Widget

### 3.1 Row 和 Column

```dart
class FlexLayoutDemo extends StatelessWidget {
  const FlexLayoutDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Row - 水平布局
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,  // 主轴对齐
          crossAxisAlignment: CrossAxisAlignment.center,      // 交叉轴对齐
          children: [
            Container(width: 60, height: 60, color: Colors.red),
            Container(width: 60, height: 60, color: Colors.green),
            Container(width: 60, height: 60, color: Colors.blue),
          ],
        ),

        const SizedBox(height: 20),

        // 使用 Expanded 实现弹性布局
        Row(
          children: [
            Expanded(
              flex: 1,
              child: Container(height: 60, color: Colors.red),
            ),
            const SizedBox(width: 8),
            Expanded(
              flex: 2,
              child: Container(height: 60, color: Colors.green),
            ),
            const SizedBox(width: 8),
            Expanded(
              flex: 1,
              child: Container(height: 60, color: Colors.blue),
            ),
          ],
        ),

        const SizedBox(height: 20),

        // Flexible 与 Expanded 的区别
        Row(
          children: [
            Flexible(
              flex: 1,
              child: Container(
                height: 60,
                color: Colors.orange,
                child: const Text('Flexible 可缩小'),
              ),
            ),
            Expanded(
              flex: 1,
              child: Container(
                height: 60,
                color: Colors.purple,
                child: const Text('Expanded 强制填满'),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
```

### 3.2 Stack 和 Positioned

```dart
class StackDemo extends StatelessWidget {
  const StackDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // 底层 - 背景图片
        Container(
          width: 300,
          height: 200,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            image: const DecorationImage(
              image: NetworkImage('https://picsum.photos/300/200'),
              fit: BoxFit.cover,
            ),
          ),
        ),

        // 中间层 - 渐变遮罩
        Container(
          width: 300,
          height: 200,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Colors.transparent, Colors.black.withOpacity(0.7)],
            ),
          ),
        ),

        // 顶层 - 文字
        const Positioned(
          bottom: 16,
          left: 16,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('图片标题', style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
              Text('副标题描述', style: TextStyle(color: Colors.white70, fontSize: 14)),
            ],
          ),
        ),

        // 角标
        Positioned(
          top: 8,
          right: 8,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.red,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Text('HOT', style: TextStyle(color: Colors.white, fontSize: 12)),
          ),
        ),
      ],
    );
  }
}
```

### 3.3 Wrap 流式布局

```dart
class WrapDemo extends StatelessWidget {
  const WrapDemo({super.key});

  @override
  Widget build(BuildContext context) {
    final tags = ['Flutter', 'Dart', 'React Native', 'Swift', 'Kotlin', 'JavaScript', 'TypeScript'];

    return Wrap(
      spacing: 8,        // 水平间距
      runSpacing: 8,     // 垂直间距
      children: tags.map((tag) => Chip(
        label: Text(tag),
        backgroundColor: Colors.blue.shade50,
      )).toList(),
    );
  }
}
```

## 四、约束机制详解

```dart
// UnconstrainedBox - 移除约束
class ConstraintDemo extends StatelessWidget {
  const ConstraintDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 父约束传递给子
        SizedBox(
          width: 200,
          height: 100,
          child: Container(
            color: Colors.blue,
            // 这里的 Container 会填满 SizedBox 的约束
          ),
        ),

        // OverflowBox - 允许溢出父约束
        ClipRect(  // 使用 ClipRect 裁剪溢出部分
          child: SizedBox(
            width: 100,
            height: 100,
            child: OverflowBox(
              maxWidth: 200,
              maxHeight: 200,
              child: Container(
                width: 150,
                height: 150,
                color: Colors.red.withOpacity(0.5),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
```

## 五、注意事项与常见陷阱

1. **避免无限约束**：Row/Column 中的子元素如果没有限制，可能导致布局异常
2. **使用 const**：尽可能使用 const 构造函数，减少 Widget 重建
3. **正确使用 Expanded/Flexible**：只在 Row、Column、Flex 中使用
4. **ListView 嵌套问题**：嵌套 ListView 需设置 `shrinkWrap: true` 和 `physics: NeverScrollableScrollPhysics()`
5. **布局调试**：使用 `debugPaintSizeEnabled = true` 可视化布局边界
