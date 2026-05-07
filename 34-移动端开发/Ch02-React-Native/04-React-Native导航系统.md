# React Native 导航系统

## 一、概念说明

React Navigation 是 React Native 官方推荐的导航库，提供了栈导航、标签导航、抽屉导航等多种导航模式，支持深层链接和状态持久化。

```bash
# 安装 React Navigation
npm install @react-navigation/native

# 安装依赖
npm install react-native-screens react-native-safe-area-context

# 安装导航器
npm install @react-navigation/native-stack
npm install @react-navigation/bottom-tabs
npm install @react-navigation/drawer
```

## 二、导航类型详解

### 2.1 栈导航（Stack Navigator）

```javascript
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

const Stack = createNativeStackNavigator();

// 导航配置
const AppNavigator = () => (
  <NavigationContainer>
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerStyle: { backgroundColor: '#3498db' },
        headerTintColor: '#fff',
        headerTitleStyle: { fontWeight: 'bold' },
      }}
    >
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{ title: '首页' }}
      />
      <Stack.Screen
        name="Detail"
        component={DetailScreen}
        options={({ route }) => ({
          title: route.params?.title || '详情',
        })}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ headerShown: false }}
      />
    </Stack.Navigator>
  </NavigationContainer>
);
```

### 2.2 页面间传参

```javascript
// 发送页面
const HomeScreen = ({ navigation }) => {
  const items = [
    { id: 1, title: '文章一', content: '内容一' },
    { id: 2, title: '文章二', content: '内容二' },
  ];

  const handlePress = (item) => {
    // 方式1: navigate 传参
    navigation.navigate('Detail', {
      itemId: item.id,
      title: item.title,
    });
  };

  const handleReplace = (item) => {
    // 方式2: replace 替换当前页面
    navigation.replace('Detail', { itemId: item.id });
  };

  return (
    <FlatList
      data={items}
      renderItem={({ item }) => (
        <TouchableOpacity onPress={() => handlePress(item)}>
          <Text>{item.title}</Text>
        </TouchableOpacity>
      )}
    />
  );
};

// 接收页面
const DetailScreen = ({ route, navigation }) => {
  const { itemId, title } = route.params;

  // 监听参数变化
  React.useEffect(() => {
    if (route.params?.itemId) {
      loadData(route.params.itemId);
    }
  }, [route.params?.itemId]);

  // 设置动态标题
  React.useLayoutEffect(() => {
    navigation.setOptions({ title: title });
  }, [navigation, title]);

  return (
    <View>
      <Text>项目ID: {itemId}</Text>
      <Button
        title="返回"
        onPress={() => navigation.goBack()}
      />
    </View>
  );
};
```

### 2.3 底部标签导航

```javascript
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/Ionicons';

const Tab = createBottomTabNavigator();

const MainTabs = () => (
  <Tab.Navigator
    screenOptions={({ route }) => ({
      tabBarIcon: ({ focused, color, size }) => {
        let iconName;
        switch (route.name) {
          case 'Home':
            iconName = focused ? 'home' : 'home-outline';
            break;
          case 'Search':
            iconName = focused ? 'search' : 'search-outline';
            break;
          case 'Notifications':
            iconName = focused ? 'notifications' : 'notifications-outline';
            break;
          case 'Profile':
            iconName = focused ? 'person' : 'person-outline';
            break;
        }
        return <Icon name={iconName} size={size} color={color} />;
      },
      tabBarActiveTintColor: '#3498db',
      tabBarInactiveTintColor: 'gray',
      tabBarStyle: {
        paddingBottom: 5,
        height: 55,
      },
    })}
  >
    <Tab.Screen name="Home" component={HomeScreen} options={{ title: '首页' }} />
    <Tab.Screen name="Search" component={SearchScreen} options={{ title: '搜索' }} />
    <Tab.Screen
      name="Notifications"
      component={NotificationScreen}
      options={{
        title: '消息',
        tabBarBadge: 3, // 显示角标
      }}
    />
    <Tab.Screen name="Profile" component={ProfileScreen} options={{ title: '我的' }} />
  </Tab.Navigator>
);
```

### 2.4 抽屉导航

```javascript
import { createDrawerNavigator } from '@react-navigation/drawer';

const Drawer = createDrawerNavigator();

const AppDrawer = () => (
  <Drawer.Navigator
    drawerContent={(props) => <CustomDrawerContent {...props} />}
    screenOptions={{
      drawerPosition: 'left',
      drawerType: 'front',
      headerShown: true,
    }}
  >
    <Drawer.Screen name="Home" component={HomeScreen} options={{ title: '首页' }} />
    <Drawer.Screen name="Settings" component={SettingsScreen} options={{ title: '设置' }} />
    <Drawer.Screen name="About" component={AboutScreen} options={{ title: '关于' }} />
  </Drawer.Navigator>
);

// 自定义抽屉内容
const CustomDrawerContent = (props) => (
  <DrawerContentScrollView {...props}>
    <View style={styles.drawerHeader}>
      <Image source={require('../assets/avatar.png')} style={styles.avatar} />
      <Text style={styles.userName}>用户名</Text>
    </View>
    <DrawerItemList {...props} />
    <DrawerItem
      label="退出登录"
      onPress={() => {/* 处理退出 */}}
      icon={({ color }) => <Icon name="log-out" size={24} color={color} />}
    />
  </DrawerContentScrollView>
);
```

## 三、导航守卫与认证流程

```javascript
// 认证流程示例
const App = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <SplashScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {isAuthenticated ? (
          // 已认证用户
          <>
            <Stack.Screen name="Main" component={MainTabs} />
            <Stack.Screen name="Detail" component={DetailScreen} />
          </>
        ) : (
          // 未认证用户
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
};
```

## 四、注意事项与常见陷阱

1. **导航器嵌套**：合理规划导航器嵌套层级，避免过深嵌套导致状态管理复杂
2. **内存管理**：及时从导航栈中移除不再需要的页面，避免内存占用过大
3. **深层链接**：配置 linking 属性支持 URL 直接打开特定页面
4. **导航事件监听**：使用 addListener 监听 focus、blur、beforeRemove 等事件
5. **类型安全**：使用 TypeScript 定义导航参数类型，避免运行时错误
