# Flutter 平台集成

## 一、概念说明

Flutter 通过 Platform Channel（方法通道、事件通道、BasicMessageChannel）实现 Dart 与原生平台（Android/iOS）的通信，用于访问平台特有的功能和 API。

```dart
// 平台通道类型
/*
1. MethodChannel      - 方法调用（最常用）
2. EventChannel       - 事件流（传感器、电池等）
3. BasicMessageChannel - 基础消息传递
*/
```

## 二、MethodChannel

### 2.1 Flutter 端

```dart
import 'package:flutter/services.dart';

class PlatformService {
  static const _channel = MethodChannel('com.myapp/platform');

  // 获取设备信息
  static Future<Map<String, dynamic>> getDeviceInfo() async {
    final result = await _channel.invokeMethod('getDeviceInfo');
    return Map<String, dynamic>.from(result);
  }

  // 获取电池电量
  static Future<int> getBatteryLevel() async {
    try {
      final level = await _channel.invokeMethod<int>('getBatteryLevel');
      return level ?? -1;
    } on PlatformException catch (e) {
      print('获取电池信息失败: ${e.message}');
      return -1;
    }
  }

  // 打开系统设置
  static Future<void> openSettings() async {
    await _channel.invokeMethod('openSettings');
  }

  // 带参数的调用
  static Future<bool> shareContent(String title, String text) async {
    final result = await _channel.invokeMethod('shareContent', {
      'title': title,
      'text': text,
    });
    return result == true;
  }
}
```

### 2.2 Android 端

```kotlin
// android/app/src/main/kotlin/com/myapp/MainActivity.kt
package com.myapp

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.myapp/platform"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getDeviceInfo" -> {
                        val deviceInfo = mapOf(
                            "brand" to android.os.Build.BRAND,
                            "model" to android.os.Build.MODEL,
                            "version" to android.os.Build.VERSION.RELEASE,
                            "sdk" to android.os.Build.VERSION.SDK_INT
                        )
                        result.success(deviceInfo)
                    }
                    "getBatteryLevel" -> {
                        val batteryManager = getSystemService(BATTERY_SERVICE) as android.os.BatteryManager
                        val level = batteryManager.getIntProperty(
                            android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY
                        )
                        result.success(level)
                    }
                    "openSettings" -> {
                        val intent = android.content.Intent(
                            android.provider.Settings.ACTION_SETTINGS
                        )
                        startActivity(intent)
                        result.success(null)
                    }
                    "shareContent" -> {
                        val title = call.argument<String>("title") ?: ""
                        val text = call.argument<String>("text") ?: ""
                        val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                            type = "text/plain"
                            putExtra(android.content.Intent.EXTRA_SUBJECT, title)
                            putExtra(android.content.Intent.EXTRA_TEXT, text)
                        }
                        startActivity(android.content.Intent.createChooser(intent, "分享"))
                        result.success(true)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}
```

### 2.3 iOS 端

```swift
// ios/Runner/AppDelegate.swift
import UIKit
import Flutter

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    private let CHANNEL = "com.myapp/platform"

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let controller = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(
            name: CHANNEL,
            binaryMessenger: controller.binaryMessenger
        )

        channel.setMethodCallHandler { (call, result) in
            switch call.method {
            case "getDeviceInfo":
                let info: [String: Any] = [
                    "brand": "Apple",
                    "model": UIDevice.current.model,
                    "version": UIDevice.current.systemVersion
                ]
                result(info)

            case "getBatteryLevel":
                UIDevice.current.isBatteryMonitoringEnabled = true
                let level = Int(UIDevice.current.batteryLevel * 100)
                result(level >= 0 ? level : FlutterError(
                    code: "UNAVAILABLE", message: "电池信息不可用", details: nil
                ))

            case "openSettings":
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
                result(nil)

            default:
                result(FlutterMethodNotImplemented)
            }
        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
}
```

## 三、EventChannel（事件流）

```dart
// Flutter 端
class BatteryStreamService {
  static const _channel = EventChannel('com.myapp/battery');

  static Stream<int> get batteryLevelStream {
    return _channel.receiveBroadcastStream().map((event) => event as int);
  }
}

// 使用
StreamBuilder<int>(
  stream: BatteryStreamService.batteryLevelStream,
  builder: (context, snapshot) {
    if (snapshot.hasData) {
      return Text('电池: ${snapshot.data}%');
    }
    return const CircularProgressIndicator();
  },
);
```

## 四、PlatformView（嵌入原生视图）

```dart
// 使用 PlatformView 嵌入原生组件
// Android
class AndroidMapView extends StatelessWidget {
  const AndroidMapView({super.key});

  @override
  Widget build(BuildContext context) {
    return AndroidView(
      viewType: 'com.myapp/mapview',
      onPlatformViewCreated: (id) {
        print('视图创建: $id');
      },
      creationParams: {
        'latitude': 39.9042,
        'longitude': 116.4074,
      },
      creationParamsCodec: const StandardMessageCodec(),
    );
  }
}

// iOS
class IosMapView extends StatelessWidget {
  const IosMapView({super.key});

  @override
  Widget build(BuildContext context) {
    return UiKitView(
      viewType: 'com.myapp/mapview',
      creationParams: const {
        'latitude': 39.9042,
        'longitude': 116.4074,
      },
      creationParamsCodec: const StandardMessageCodec(),
    );
  }
}
```

## 五、常用插件

```yaml
# 常用平台插件
dependencies:
  # 设备功能
  camera: ^0.10.0
  image_picker: ^1.0.0
  permission_handler: ^10.4.0
  url_launcher: ^6.1.0
  share_plus: ^7.0.0

  # 传感器
  sensors_plus: ^4.0.0
  geolocator: ^10.0.0

  # 通信
  flutter_local_notifications: ^15.0.0
  firebase_core: ^2.15.0
  firebase_messaging: ^14.6.0

  # 文件系统
  path_provider: ^2.1.0
  file_picker: ^5.3.0
```

## 六、注意事项与常见陷阱

1. **线程问题**：Platform Channel 在主线程调用，耗时操作应在原生端异步处理
2. **类型安全**：Dart 和原生端的类型需要正确映射，使用 StandardMessageCodec
3. **错误处理**：原生端应使用 result.error() 返回错误信息
4. **插件选择**：优先使用 pub.dev 上活跃维护的插件
5. **平台差异**：某些功能只在特定平台可用，需要条件导入
