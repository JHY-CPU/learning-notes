# React Native 推送通知

## 一、概念说明

推送通知是移动应用与用户保持互动的重要手段，即使应用在后台或关闭状态也能向用户发送消息。移动端推送通知主要依赖 APNs（iOS）和 FCM（Android）。

```javascript
// 安装推送通知库
// npm install @react-native-firebase/app @react-native-firebase/messaging
// npm install @notifee/react-native

import messaging from '@react-native-firebase/messaging';
import notifee from '@notifee/react-native';
```

## 二、推送通知实现

### 2.1 权限申请

```javascript
// 请求通知权限
const requestNotificationPermission = async () => {
  // iOS 需要明确请求权限
  const authStatus = await messaging().requestPermission();
  const enabled =
    authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
    authStatus === messaging.AuthorizationStatus.PROVISIONAL;

  if (enabled) {
    console.log('通知权限已授权:', authStatus);
    // 获取 FCM Token
    const token = await messaging().getToken();
    console.log('FCM Token:', token);
    // 将 Token 发送到服务器
    await sendTokenToServer(token);
  }

  return enabled;
};

// 监听 Token 刷新
messaging().onTokenRefresh(async (token) => {
  console.log('Token 刷新:', token);
  await sendTokenToServer(token);
});
```

### 2.2 消息处理

```javascript
// 前台消息处理
messaging().onMessage(async (remoteMessage) => {
  console.log('前台收到消息:', remoteMessage);
  // 显示本地通知
  await displayLocalNotification(remoteMessage);
});

// 后台/退出消息处理
messaging().setBackgroundMessageHandler(async (remoteMessage) => {
  console.log('后台收到消息:', remoteMessage);
  // 处理后台消息
  await processBackgroundMessage(remoteMessage);
});

// 用户点击通知
messaging().onNotificationOpenedApp((remoteMessage) => {
  console.log('用户点击通知打开应用:', remoteMessage);
  // 导航到对应页面
  handleNotificationNavigation(remoteMessage);
});

// 应用从通知冷启动
messaging().getInitialNotification().then((remoteMessage) => {
  if (remoteMessage) {
    console.log('应用通过通知冷启动:', remoteMessage);
    handleNotificationNavigation(remoteMessage);
  }
});
```

### 2.3 本地通知（Notifee）

```javascript
import notifee, { AndroidImportance, TriggerType } from '@notifee/react-native';

// 创建通知渠道（Android 必须）
const createNotificationChannel = async () => {
  const channelId = await notifee.createChannel({
    id: 'default',
    name: '默认通知',
    importance: AndroidImportance.HIGH,
    sound: 'default',
    vibration: true,
  });
  return channelId;
};

// 显示即时通知
const displayLocalNotification = async (remoteMessage) => {
  const channelId = await createNotificationChannel();

  await notifee.displayNotification({
    title: remoteMessage.notification?.title || '新消息',
    body: remoteMessage.notification?.body || '',
    data: remoteMessage.data,
    android: {
      channelId,
      importance: AndroidImportance.HIGH,
      pressAction: { id: 'default' },
      smallIcon: 'ic_notification',
      largeIcon: 'https://example.com/avatar.png',
    },
    ios: {
      sound: 'default',
      badgeCount: 1,
    },
  });
};

// 定时通知
const scheduleNotification = async (title, body, triggerDate) => {
  const channelId = await createNotificationChannel();

  await notifee.createTriggerNotification(
    {
      title,
      body,
      android: { channelId },
    },
    {
      type: TriggerType.TIMESTAMP,
      timestamp: triggerDate.getTime(),
    }
  );
};

// 取消所有通知
const cancelAllNotifications = async () => {
  await notifee.cancelAllNotifications();
};
```

### 2.4 通知数据处理

```javascript
// 统一的通知处理
const NotificationService = {
  // 初始化
  init: async () => {
    // 请求权限
    await requestNotificationPermission();

    // 设置通知处理回调
    messaging().onMessage(NotificationService.handleForeground);
    messaging().setBackgroundMessageHandler(NotificationService.handleBackground);
    messaging().onNotificationOpenedApp(NotificationService.handleOpen);
  },

  // 前台处理
  handleForeground: async (message) => {
    await displayLocalNotification(message);
  },

  // 后台处理
  handleBackground: async (message) => {
    // 存储消息供后续处理
    await Storage.set('pending_notification', message);
  },

  // 用户打开通知
  handleOpen: (message) => {
    const { type, id } = message.data || {};
    switch (type) {
      case 'chat':
        NavigationService.navigate('Chat', { chatId: id });
        break;
      case 'order':
        NavigationService.navigate('OrderDetail', { orderId: id });
        break;
      default:
        NavigationService.navigate('Notifications');
    }
  },
};
```

## 三、消息格式

```json
// FCM 消息格式
{
  "to": "device_fcm_token",
  "notification": {
    "title": "新消息",
    "body": "你收到了一条新消息",
    "sound": "default"
  },
  "data": {
    "type": "chat",
    "id": "12345",
    "sender": "张三"
  },
  "priority": "high",
  "ttl": 3600
}
```

## 四、注意事项与常见陷阱

1. **平台差异**：iOS 必须明确请求权限，Android 8+ 必须创建通知渠道
2. **Token 管理**：FCM Token 可能变化，需要监听刷新事件并更新服务器
3. **后台限制**：Android 后台执行有限制，避免在后台处理中做耗时操作
4. **通知分组**：Android 支持通知分组，大量通知时应合理分组
5. **测试环境**：推送通知需要真机测试，模拟器无法接收远程推送
