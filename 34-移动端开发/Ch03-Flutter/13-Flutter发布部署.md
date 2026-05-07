# Flutter 发布部署

## 一、概念说明

Flutter 应用发布涉及应用签名、构建优化和应用商店提交。iOS 和 Android 平台的发布流程各有不同。

```bash
# 构建发布版本
flutter build apk --release           # Android APK
flutter build appbundle --release     # Android AAB (推荐)
flutter build ios --release           # iOS
flutter build web --release           # Web
flutter build macos --release         # macOS
```

## 二、Android 发布

### 2.1 签名配置

```bash
# 生成密钥库
keytool -genkey -v -keystore ~/my-release-key.jks \
  -keyalg RSA -keysize 2048 -validity 10000 -alias my-key-alias
```

```properties
# android/key.properties
storePassword=your_store_password
keyPassword=your_key_password
keyAlias=my-key-alias
storeFile=/path/to/my-release-key.jks
```

```groovy
// android/app/build.gradle
def keystoreProperties = new Properties()
def keystorePropertiesFile = rootProject.file('key.properties')
keystoreProperties.load(new FileInputStream(keystorePropertiesFile))

android {
    signingConfigs {
        release {
            keyAlias keystoreProperties['keyAlias']
            keyPassword keystoreProperties['keyPassword']
            storeFile file(keystoreProperties['storeFile'])
            storePassword keystoreProperties['storePassword']
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}
```

### 2.2 版本管理

```yaml
# pubspec.yaml
version: 1.2.3+45
#        ^^^ ^  <- 版本号 + 构建号
```

```groovy
// android/app/build.gradle
android {
    defaultConfig {
        versionCode flutterVersionCode.toInteger()  // 内部版本号
        versionName flutterVersionName               // 显示版本号
    }
}
```

## 三、iOS 发布

### 3.1 Xcode 配置

```bash
# 1. 在 Apple Developer 创建 App ID
# 2. 创建证书和 Provisioning Profile
# 3. 在 Xcode 中配置签名

# 更新 CocoaPods
cd ios && pod install && cd ..

# 构建 Archive
# 在 Xcode 中: Product -> Archive
```

```xml
<!-- ios/Runner/Info.plist -->
<key>CFBundleDisplayName</key>
<string>我的应用</string>
<key>CFBundleShortVersionString</key>
<string>1.2.3</string>
<key>CFBundleVersion</key>
<string>45</string>

<!-- 权限描述 -->
<key>NSCameraUsageDescription</key>
<string>需要访问相机拍照</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>需要访问相册选择图片</string>
```

## 四、Fastlane 自动化

```ruby
# fastlane/Fastfile
platform :ios do
  desc "发布到 TestFlight"
  lane :beta do
    increment_build_number
    build_app(
      workspace: "Runner.xcworkspace",
      scheme: "Runner",
      export_method: "app-store"
    )
    upload_to_testflight
  end
end

platform :android do
  desc "发布到 Google Play"
  lane :production do
    flutter_build_appbundle
    upload_to_play_store(
      track: "production",
      aab: "build/app/outputs/bundle/release/app-release.aab"
    )
  end
end
```

```bash
# 运行
cd ios && fastlane beta
cd android && fastlane production
```

## 五、CI/CD 配置

```yaml
# .github/workflows/release.yml
name: Flutter Release
on:
  push:
    tags: ['v*']

jobs:
  build-android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
        with:
          flutter-version: '3.16.0'
      - run: flutter pub get
      - run: flutter build appbundle --release
      - uses: actions/upload-artifact@v3
        with:
          name: android-release
          path: build/app/outputs/bundle/release/app-release.aab

  build-ios:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
      - run: cd ios && pod install
      - run: flutter build ios --release --no-codesign
```

## 六、应用瘦身

```yaml
# 减小包体积
flutter build appbundle --release --shrink

# 使用 --split-per-abi 生成多个 APK
flutter build apk --release --split-per-abi

# 排除不需要的资源
flutter:
  assets:
    - assets/images/
  # 不要包含测试资源
```

## 七、注意事项与常见陷阱

1. **版本号管理**：每次发布必须递增构建号（versionCode/CFBundleVersion）
2. **Proguard 混淆**：启用混淆后需充分测试，某些反射操作可能被破坏
3. **iOS 审核**：Apple 审核严格，确保符合 App Store Review Guidelines
4. **密钥安全**：签名密钥不要提交到 Git，使用环境变量或安全存储
5. **应用图标**：使用 flutter_launcher_icons 生成各平台图标
