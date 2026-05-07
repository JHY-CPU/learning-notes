# Flutter 列表与滚动

## 一、概念说明

Flutter 提供了多种滚动组件，用于高效展示大量数据。ListView 是最常用的列表组件，支持懒加载优化性能。GridView 用于网格布局，CustomScrollView 支持复杂的滚动效果。

```dart
// 滚动组件概览
/*
ListView       - 线性列表
GridView       - 网格列表
SingleChildScrollView - 单子滚动
CustomScrollView      - 自定义滚动 (Sliver)
PageView              - 页面滑动
NestedScrollView      - 嵌套滚动
*/
```

## 二、ListView 详解

### 2.1 基础用法

```dart
// 四种构造方式

// 1. ListView() - 适合少量固定数据
ListView(
  children: [
    ListTile(title: Text('项目1')),
    ListTile(title: Text('项目2')),
    ListTile(title: Text('项目3')),
  ],
);

// 2. ListView.builder() - 适合大量数据（懒加载）
ListView.builder(
  itemCount: 1000,
  itemBuilder: (context, index) {
    return ListTile(
      leading: CircleAvatar(child: Text('$index')),
      title: Text('项目 $index'),
      subtitle: Text('这是第 $index 个项目的描述'),
    );
  },
);

// 3. ListView.separated() - 带分隔线
ListView.separated(
  itemCount: 100,
  separatorBuilder: (context, index) => const Divider(height: 1),
  itemBuilder: (context, index) {
    return ListTile(title: Text('项目 $index'));
  },
);

// 4. ListView.custom() - 完全自定义
ListView.custom(
  childrenDelegate: SliverChildBuilderDelegate(
    (context, index) => ListTile(title: Text('项目 $index')),
    childCount: 100,
  ),
);
```

### 2.2 下拉刷新与上拉加载

```dart
class RefreshLoadMoreList extends StatefulWidget {
  const RefreshLoadMoreList({super.key});

  @override
  State<RefreshLoadMoreList> createState() => _RefreshLoadMoreListState();
}

class _RefreshLoadMoreListState extends State<RefreshLoadMoreList> {
  final List<String> _items = [];
  bool _isLoading = false;
  int _page = 1;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    await Future.delayed(const Duration(seconds: 1));
    setState(() {
      _items.addAll(List.generate(20, (i) => '项目 ${_items.length + i + 1}'));
    });
  }

  Future<void> _onRefresh() async {
    setState(() {
      _items.clear();
      _page = 1;
    });
    await _loadData();
  }

  Future<void> _loadMore() async {
    if (_isLoading) return;
    setState(() => _isLoading = true);
    _page++;
    await _loadData();
    setState(() => _isLoading = false);
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: _onRefresh,
      child: ListView.builder(
        itemCount: _items.length + 1,
        itemBuilder: (context, index) {
          if (index == _items.length) {
            return _isLoading
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: CircularProgressIndicator(),
                    ),
                  )
                : const SizedBox.shrink();
          }
          return ListTile(
            leading: CircleAvatar(child: Text('${index + 1}')),
            title: Text(_items[index]),
            trailing: const Icon(Icons.chevron_right),
          );
        },
      ),
    );
  }
}
```

## 三、GridView

```dart
// GridView 基础
class GridDemo extends StatelessWidget {
  const GridDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      padding: const EdgeInsets.all(8),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,         // 列数
        childAspectRatio: 1.5,     // 宽高比
        crossAxisSpacing: 8,       // 水平间距
        mainAxisSpacing: 8,        // 垂直间距
      ),
      itemCount: 20,
      itemBuilder: (context, index) {
        return Card(
          elevation: 2,
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.image, size: 48, color: Colors.blue),
                const SizedBox(height: 8),
                Text('项目 ${index + 1}'),
              ],
            ),
          ),
        );
      },
    );
  }
}

// 动态列数（根据屏幕宽度）
class ResponsiveGrid extends StatelessWidget {
  const ResponsiveGrid({super.key});

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final crossAxisCount = screenWidth ~/ 150; // 每个格子最小150px

    return GridView.builder(
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount.clamp(2, 6),
        childAspectRatio: 1,
      ),
      itemCount: 20,
      itemBuilder: (context, index) => Card(child: Center(child: Text('$index'))),
    );
  }
}
```

## 四、CustomScrollView 与 Sliver

```dart
class SliverDemo extends StatelessWidget {
  const SliverDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      slivers: [
        // 折叠式 AppBar
        SliverAppBar(
          expandedHeight: 200,
          floating: false,
          pinned: true,
          flexibleSpace: FlexibleSpaceBar(
            title: const Text('Sliver 示例'),
            background: Image.network(
              'https://picsum.photos/400/200',
              fit: BoxFit.cover,
            ),
          ),
        ),

        // 网格列表
        SliverGrid(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3,
            mainAxisSpacing: 4,
            crossAxisSpacing: 4,
          ),
          delegate: SliverChildBuilderDelegate(
            (context, index) => Container(
              color: Colors.primaries[index % Colors.primaries.length],
              child: Center(child: Text('$index')),
            ),
            childCount: 12,
          ),
        ),

        // 列表
        SliverList(
          delegate: SliverChildBuilderDelegate(
            (context, index) => ListTile(
              leading: const Icon(Icons.star),
              title: Text('项目 $index'),
            ),
            childCount: 30,
          ),
        ),

        // 固定高度项
        SliverFixedExtentList(
          itemExtent: 60,
          delegate: SliverChildBuilderDelegate(
            (context, index) => ListTile(title: Text('固定高度 $index')),
            childCount: 20,
          ),
        ),
      ],
    );
  }
}
```

## 五、滚动监听

```dart
class ScrollListenerDemo extends StatefulWidget {
  const ScrollListenerDemo({super.key});

  @override
  State<ScrollListenerDemo> createState() => _ScrollListenerDemoState();
}

class _ScrollListenerDemoState extends State<ScrollListenerDemo> {
  final ScrollController _controller = ScrollController();
  bool _showBackToTop = false;

  @override
  void initState() {
    super.initState();
    _controller.addListener(() {
      setState(() {
        _showBackToTop = _controller.offset > 300;
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView.builder(
        controller: _controller,
        itemCount: 100,
        itemBuilder: (context, index) => ListTile(title: Text('项目 $index')),
      ),
      floatingActionButton: _showBackToTop
          ? FloatingActionButton(
              onPressed: () {
                _controller.animateTo(
                  0,
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut,
                );
              },
              child: const Icon(Icons.arrow_upward),
            )
          : null,
    );
  }
}
```

## 六、注意事项与常见陷阱

1. **性能优化**：大量数据始终使用 `builder` 构造，避免一次性创建所有子项
2. **itemExtent 优化**：固定高度的列表项设置 `itemExtent`，避免测量开销
3. **ScrollController 释放**：在 dispose 中调用 `_controller.dispose()`
4. **嵌套滚动冲突**：使用 `NeverScrollableScrollPhysics()` 禁用内部列表的滚动
5. **键盘弹出时的滚动**：使用 `resizeToAvoidBottomInset` 配合处理
