# React Native 原生模块

## 一、概念说明

原生模块是 React Native 与平台原生代码（Java/Kotlin/Objective-C/Swift）之间的桥梁。当内置模块无法满足需求时，需要编写自定义原生模块来访问设备硬件或平台特有功能。

```javascript
// React Native 新架构 - Turbo Modules
// 使用 Codegen 自动生成类型安全的原生模块接口
import { NativeModules } from 'react-native';

const { CalendarModule } = NativeModules;

// 调用原生方法
const createEvent = async () => {
  try {
    const eventId = await CalendarModule.createEvent(
      '会议',
      '2024-01-15T10:00:00',
      '北京'
    );
    console.log('创建事件ID:', eventId);
  } catch (error) {
    console.error('创建事件失败:', error);
  }
};
```

## 二、原生模块开发

### 2.1 Android 原生模块（Java）

```java
// android/app/src/main/java/com/myapp/CalendarModule.java
package com.myapp;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReadableMap;

public class CalendarModule extends ReactContextBaseJavaModule {
    CalendarModule(ReactApplicationContext context) {
        super(context);
    }

    @Override
    public String getName() {
        return "CalendarModule";
    }

    @ReactMethod
    public void createEvent(String name, String location, Promise promise) {
        try {
            // 执行原生操作
            String eventId = "evt_" + System.currentTimeMillis();
            // ... 实际创建日历事件的逻辑
            promise.resolve(eventId);
        } catch (Exception e) {
            promise.reject("CREATE_EVENT_ERROR", e.getMessage());
        }
    }

    @ReactMethod
    public void getDeviceInfo(Promise promise) {
        try {
            WritableMap info = new WritableNativeMap();
            info.putString("brand", android.os.Build.BRAND);
            info.putString("model", android.os.Build.MODEL);
            info.putString("systemVersion", android.os.Build.VERSION.RELEASE);
            promise.resolve(info);
        } catch (Exception e) {
            promise.reject("DEVICE_INFO_ERROR", e.getMessage());
        }
    }

    // 常量导出
    @Override
    public Map<String, Object> getConstants() {
        final Map<String, Object> constants = new HashMap<>();
        constants.put("VERSION", "1.0.0");
        return constants;
    }
}
```

### 2.2 iOS 原生模块（Swift）

```swift
// ios/CalendarModule.swift
import Foundation
import React

@objc(CalendarModule)
class CalendarModule: NSObject {

  @objc
  static func requiresMainQueueSetup() -> Bool {
    return true
  }

  @objc
  func createEvent(_ name: String,
                   location: String,
                   resolver resolve: @escaping RCTPromiseResolveBlock,
                   rejecter reject: @escaping RCTPromiseRejectBlock) {
    // 执行原生操作
    let eventId = "evt_\(Date().timeIntervalSince1970)"
    // ... 实际创建日历事件的逻辑
    resolve(eventId)
  }

  @objc
  func getDeviceInfo(_ resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    let info: [String: Any] = [
      "brand": "Apple",
      "model": UIDevice.current.model,
      "systemVersion": UIDevice.current.systemVersion
    ]
    resolve(info)
  }
}
```

### 2.3 JavaScript 桥接封装

```javascript
// modules/CalendarModule.js
import { NativeModules, Platform } from 'react-native';

const { CalendarModule } = NativeModules;

const Calendar = {
  // 创建日历事件
  createEvent: (name, location) => {
    return CalendarModule.createEvent(name, location);
  },

  // 获取设备信息
  getDeviceInfo: () => {
    return CalendarModule.getDeviceInfo();
  },

  // 平台特定调用
  openSettings: () => {
    if (Platform.OS === 'ios') {
      CalendarModule.openIOSSettings();
    } else {
      CalendarModule.openAndroidSettings();
    }
  },

  // 常量
  VERSION: CalendarModule.VERSION,
};

export default Calendar;
```

## 三、原生 UI 组件

```java
// Android 自定义 View Manager
// android/app/src/main/java/com/myapp/MapViewManager.java
package com.myapp;

import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.annotations.ReactProp;

public class MapViewManager extends SimpleViewManager<MapView> {
    public static final String REACT_CLASS = "RCTMapView";

    @Override
    public String getName() {
        return REACT_CLASS;
    }

    @Override
    protected MapView createViewInstance(ThemedReactContext context) {
        return new MapView(context);
    }

    @ReactProp(name = "latitude")
    public void setLatitude(MapView view, double latitude) {
        view.setLatitude(latitude);
    }

    @ReactProp(name = "longitude")
    public void setLongitude(MapView view, double longitude) {
        view.setLongitude(longitude);
    }

    @ReactProp(name = "zoom")
    public void setZoom(MapView view, float zoom) {
        view.setZoom(zoom);
    }
}
```

```javascript
// 使用原生组件
import { requireNativeComponent } from 'react-native';

const NativeMapView = requireNativeComponent('RCTMapView');

const MapScreen = () => (
  <NativeMapView
    style={{ flex: 1 }}
    latitude={39.9042}
    longitude={116.4074}
    zoom={12}
  />
);
```

## 四、注意事项与常见陷阱

1. **线程管理**：原生模块方法在后台线程执行，更新 UI 需切换到主线程
2. **内存泄漏**：注意原生对象的生命周期，避免持有 React Context 的强引用
3. **类型映射**：JS 和原生类型需要正确映射，复杂对象使用 ReadableMap/NSDictionary
4. **错误处理**：Promise reject 需要提供错误码和描述，便于 JS 端排查
5. **版本兼容**：注意 React Native 版本变化对原生模块 API 的影响
