# React Native 部署与发布

## 一、概念说明

React Native 应用的发布涉及代码签名、构建优化、应用商店提交等流程。iOS 和 Android 平台的发布流程和要求各不相同。

```bash
# 构建发布版本

# Android Release APK/AAB
cd android
./gradlew assembleRelease    # 生成 APK
./gradlew bundleRelease      # 生成 AAB (推荐)

# iOS Release
# 在 Xcode 中配置签名后
xcodebuild -workspace ios/MyApp.xcworkspace \
  -scheme MyApp \
  -configuration Release \
  -derivedDataPath ios/build
```

## 二、Android 发布

### 2.1 签名配置

```gradle
// android/app/build.gradle
android {
    signingConfigs {
        release {
            storeFile file('my-release-key.keystore')
            storePassword 'your_store_password'
            keyAlias 'my-key-alias'
            keyPassword 'your_key_password'
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

```bash
# 生成签名密钥
keytool -genkeypair -v -storetype PKCS12 \
  -keystore my-release-key.keystore \
  -alias my-key-alias \
  -keyalg RSA -keysize 2048 \
  -validity 10000
```

### 2.2 Proguard 配置

```proguard
// android/app/proguard-rules.pro
# React Native 核心
-keep class com.facebook.react.** { *; }
-keep class com.facebook.hermes.** { *; }
-keep class com.facebook.jni.** { *; }

# 保留原生模块
-keepclassmembers class * extends com.facebook.react.bridge.ReactContextBaseJavaModule {
    @com.facebook.react.bridge.ReactMethod *;
}

# 保留第三方库
-keep class com.swmansion.** { *; }
-keep class com.reactnativecommunity.** { *; }
```

### 2.3 版本管理

```gradle
// android/app/build.gradle
android {
    defaultConfig {
        applicationId "com.myapp"
        versionCode 1          // 内部版本号，每次发布递增
        versionName "1.0.0"    // 显示版本号
        minSdkVersion 21
        targetSdkVersion 33
    }
}
```

## 三、iOS 发布

### 3.1 签名与配置

```bash
# 1. 在 Apple Developer 创建 App ID 和证书
# 2. 创建 Provisioning Profile
# 3. 在 Xcode 中配置签名

# 自动签名
# Xcode -> General -> Signing & Capabilities
# 勾选 "Automatically manage signing"

# 手动签名
# 选择对应的 Team、Provisioning Profile 和证书
```

### 3.2 Info.plist 配置

```xml
<!-- ios/MyApp/Info.plist -->
<key>CFBundleDisplayName</key>
<string>我的应用</string>
<key>CFBundleShortVersionString</key>
<string>1.0.0</string>
<key>CFBundleVersion</key>
<string>1</string>

<!-- 权限描述 -->
<key>NSCameraUsageDescription</key>
<string>需要访问相机以拍照上传</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>需要访问相册以选择图片</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>需要获取位置信息以提供本地服务</string>
```

### 3.3 构建与上传

```bash
# 使用 Xcode 构建 Archive
# Product -> Archive

# 或使用命令行
xcodebuild -workspace ios/MyApp.xcworkspace \
  -scheme MyApp \
  -configuration Release \
  -archivePath build/MyApp.xcarchive \
  archive

# 导出 IPA
xcodebuild -exportArchive \
  -archivePath build/MyApp.xcarchive \
  -exportOptionsPlist ExportOptions.plist \
  -exportPath build
```

## 四、自动化发布

### 4.1 Fastlane 配置

```ruby
# fastlane/Fastfile
platform :ios do
  desc "发布到 TestFlight"
  lane :beta do
    increment_build_number
    build_app(scheme: "MyApp")
    upload_to_testflight
  end

  desc "发布到 App Store"
  lane :release do
    increment_build_number
    build_app(scheme: "MyApp")
    upload_to_app_store(
      submit_for_review: true,
      automatic_release: true
    )
  end
end

platform :android do
  desc "发布到 Google Play"
  lane :release do
    gradle(task: "bundleRelease")
    upload_to_play_store(track: "production")
  end
end
```

### 4.2 CI/CD 流程

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  release-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-java@v3
        with:
          distribution: 'zulu'
          java-version: '11'
      - run: npm ci
      - run: cd android && ./gradlew bundleRelease
      - uses: r0adkll/upload-google-play@v1
        with:
          releaseFiles: android/app/build/outputs/bundle/release/app-release.aab
          packageName: com.myapp
          status: completed

  release-ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm ci
      - run: cd ios && pod install
      - run: fastlane beta
```

### 4.3 CodePush 热更新

```javascript
// 集成 CodePush
import codePush from 'react-native-code-push';

const codePushOptions = {
  checkFrequency: codePush.CheckFrequency.ON_APP_RESUME,
  installMode: codePush.InstallMode.IMMEDIATE,
};

class App extends React.Component {
  componentDidMount() {
    codePush.sync({
      updateDialog: {
        title: '更新提示',
        optionalUpdateMessage: '发现新版本，是否更新？',
        optionalInstallButtonLabel: '更新',
        optionalIgnoreButtonLabel: '忽略',
      },
      installMode: codePush.InstallMode.IMMEDIATE,
    });
  }

  render() {
    return <MainNavigator />;
  }
}

export default codePush(codePushOptions)(App);
```

```bash
# 发布热更新
code-push release-react MyApp ios -d Production
code-push release-react MyApp android -d Production
```

## 五、注意事项与常见陷阱

1. **版本号管理**：每次发布必须递增 versionCode/CFBundleVersion
2. **iOS 审核**：注意 Apple 审核指南，避免使用私有 API 和热更新违规
3. **Proguard 混淆**：开启混淆后需测试所有功能，确保没有被误混淆
4. **大包体积**：注意 Android 的 MultiDex 和 iOS 的 App Thinning
5. **证书过期**：定期检查签名证书和 Provisioning Profile 的有效期
