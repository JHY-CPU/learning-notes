# React Native 性能优化

## 一、概念说明

React Native 应用性能优化涉及渲染优化、内存管理、启动速度和包体积等多个方面。良好的性能优化策略能确保应用流畅运行，提供接近原生的用户体验。

```javascript
// 性能监控基础
import { Performance } from 'react-native';

const measurePerformance = (label, fn) => {
  const start = performance.now();
  const result = fn();
  const end = performance.now();
  console.log(`${label}: ${(end - start).toFixed(2)}ms`);
  return result;
};
```

## 二、渲染优化

### 2.1 减少重渲染

```javascript
// 使用 React.memo 缓存组件
const ListItem = React.memo(({ item, onPress }) => {
  console.log('ListItem 渲染:', item.id);
  return (
    <TouchableOpacity onPress={() => onPress(item.id)}>
      <Text>{item.title}</Text>
    </TouchableOpacity>
  );
}, (prevProps, nextProps) => {
  // 自定义比较函数
  return prevProps.item.id === nextProps.item.id &&
         prevProps.item.title === nextProps.item.title;
});

// 使用 useMemo 缓存计算结果
const ExpensiveList = ({ data, filter }) => {
  const filteredData = useMemo(() => {
    console.log('重新计算过滤数据');
    return data.filter(item => item.category === filter);
  }, [data, filter]);

  const sortedData = useMemo(() => {
    return [...filteredData].sort((a, b) => b.date - a.date);
  }, [filteredData]);

  return (
    <FlatList
      data={sortedData}
      renderItem={({ item }) => <ListItem item={item} />}
    />
  );
};

// 使用 useCallback 缓存函数
const ParentComponent = () => {
  const [items, setItems] = useState([]);

  const handlePress = useCallback((id) => {
    console.log('点击项目:', id);
    // 处理点击逻辑
  }, []);

  const handleDelete = useCallback((id) => {
    setItems(prev => prev.filter(item => item.id !== id));
  }, []);

  return (
    <FlatList
      data={items}
      renderItem={({ item }) => (
        <ListItem
          item={item}
          onPress={handlePress}
          onDelete={handleDelete}
        />
      )}
    />
  );
};
```

### 2.2 列表性能优化

```javascript
// FlatList 性能优化配置
const OptimizedFlatList = ({ data }) => (
  <FlatList
    data={data}
    renderItem={renderItem}
    keyExtractor={keyExtractor}
    // 关键性能配置
    initialNumToRender={10}       // 首次渲染数量
    maxToRenderPerBatch={10}      // 每批渲染数量
    windowSize={5}                // 可见区域外渲染的范围
    removeClippedSubviews={true}  // 移除不可见子视图
    // 固定高度优化
    getItemLayout={(data, index) => ({
      length: ITEM_HEIGHT,
      offset: ITEM_HEIGHT * index,
      index,
    })}
    // 性能相关的回调
    onEndReachedThreshold={0.5}
    updateCellsBatchingPeriod={50}
    // 使用 FlashList（更高性能）
  />
);

// 使用 FlashList（Shopify 开源）
import { FlashList } from '@shopify/flash-list';

const UltraFastList = ({ data }) => (
  <FlashList
    data={data}
    renderItem={({ item }) => <ListItem item={item} />}
    estimatedItemSize={80}
  />
);
```

### 2.3 图片优化

```javascript
// 图片优化策略
import FastImage from 'react-native-fast-image';

const OptimizedImage = ({ uri }) => (
  <FastImage
    source={{
      uri,
      priority: FastImage.priority.normal,
      cache: FastImage.cacheControl.immutable,
    }}
    style={{ width: 200, height: 200 }}
    resizeMode={FastImage.resizeMode.cover}
  />
);

// 图片预加载
const preloadImages = (urls) => {
  FastImage.preload(urls.map(url => ({ uri: url })));
};

// 列表图片懒加载
const LazyImage = ({ uri, placeholder }) => {
  const [loaded, setLoaded] = useState(false);

  return (
    <FastImage
      source={{ uri }}
      style={{ width: 100, height: 100 }}
      onLoad={() => setLoaded(true)}
    />
  );
};
```

## 三、内存优化

```javascript
// 避免内存泄漏
const DataScreen = () => {
  const [data, setData] = useState(null);
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;

    const loadData = async () => {
      const result = await fetchData();
      // 检查组件是否仍然挂载
      if (isMounted.current) {
        setData(result);
      }
    };

    loadData();

    return () => {
      // 清理函数
      isMounted.current = false;
    };
  }, []);

  return <View>{data && <DataDisplay data={data} />}</View>;
};

// 使用 WeakRef 处理大对象
const processLargeData = (data) => {
  const ref = new WeakRef(data);
  // 处理数据
  const result = heavyProcessing(data);
  // 原始数据可以被垃圾回收
  return result;
};
```

## 四、启动优化

```javascript
// 延迟加载非关键模块
const HeavyScreen = React.lazy(() => import('./HeavyScreen'));

// 使用 InteractionManager 延迟执行
import { InteractionManager } from 'react-native';

const OptimizedScreen = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    // 等待动画和交互完成后再执行
    InteractionManager.runAfterInteractions(async () => {
      const result = await loadHeavyData();
      setData(result);
    });
  }, []);

  return data ? <DataView data={data} /> : <Loading />;
};
```

## 五、注意事项与常见陷阱

1. **不要在 render 中创建新对象/函数**：这会导致子组件不必要的重渲染
2. **避免匿名函数**：FlatList renderItem 中使用匿名函数会影响性能
3. **合理使用 native driver**：动画使用 `useNativeDriver: true` 减少 JS 线程负担
4. **监控 JS 线程帧率**：使用 Flipper 或性能面板监控帧率，低于 60fps 需要优化
5. **包体积优化**：使用 Proguard 和 Hermes 引擎减小包体积
