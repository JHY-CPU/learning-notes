# React Native 新架构

## 一、概念说明

React Native 新架构（0.68+）引入了 Fabric 渲染器、TurboModules、Codegen 和 Hermes 引擎等核心改进，大幅提升了性能和开发体验。

```javascript
// 新架构核心组件
/*
旧架构:
JS Thread -> Bridge -> Native Modules -> Shadow Thread -> UI Thread
(序列化通信，性能瓶颈)

新架构:
JS Thread -> JSI (JavaScript Interface) -> TurboModules/Fabric
(直接内存访问，同步调用)
*/
```

## 二、新架构核心特性

### 2.1 JSI（JavaScript Interface）

```javascript
// JSI 允许 JS 直接持有 C++ 对象的引用
// 不再需要 Bridge 的序列化/反序列化
// 同步调用，大幅提升性能

// 旧架构（异步，经过序列化）
NativeModules.SomeModule.someMethod(data);

// 新架构（同步，直接调用）
global.__SomeModule_someMethod(data);
```

### 2.2 TurboModules

```javascript
// TurboModules - 按需加载的原生模块
// 只在首次调用时初始化，减少启动时间

// native-specs/NativeCalculator.ts
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  add(a: number, b: number): number;
  multiply(a: number, b: number): Promise<number>;
  getConstants(): {
    PI: number;
    E: number;
  };
}

export default TurboModuleRegistry.getEnforcing<Spec>('Calculator');
```

```javascript
// 使用 TurboModule
import Calculator from './native-specs/NativeCalculator';

const result = Calculator.add(1, 2);
console.log('PI:', Calculator.getConstants().PI);
```

### 2.3 Fabric 渲染器

```javascript
// Fabric - 新的渲染系统
// 支持同步布局、并发渲染、跨平台一致性

// Fabric 组件声明
// native-specs/NativeMapView.ts
import type { ViewProps } from 'react-native';
import type { HostComponent } from 'react-native';
import codegenNativeComponent from 'react-native/Libraries/Utilities/codegenNativeComponent';

interface NativeProps extends ViewProps {
  latitude: number;
  longitude: number;
  zoomLevel?: number;
  onRegionChange?: (event: { nativeEvent: { latitude: number; longitude: number } }) => void;
}

export default codegenNativeComponent<NativeProps>('MapView') as HostComponent<NativeProps>;
```

### 2.4 Codegen（代码生成）

```json
// package.json 配置 Codegen
{
  "name": "my-turbo-module",
  "codegenConfig": {
    "name": "MyTurboModule",
    "type": "modules",
    "jsSrcsDir": "src",
    "android": {
      "javaPackageName": "com.myapp"
    }
  }
}
```

```bash
# 运行 Codegen
cd ios && pod install  # iOS 自动生成
cd android && ./gradlew generateCodegenArtifactsFromSchema  # Android
```

### 2.5 Hermes 引擎

```javascript
// Hermes 是 React Native 默认的 JS 引擎
// 优势：启动更快、内存占用更低、包体积更小

// 启用 Hermes (新版本默认启用)
// android/app/build.gradle
project.ext.react = [
    enableHermes: true,
]

// ios/Podfile
use_react_native!(
  :hermes_enabled => true
)
```

## 三、迁移到新架构

```javascript
// 1. 更新依赖
// 确保所有第三方库支持新架构

// 2. 启用新架构
// android/gradle.properties
newArchEnabled=true

// ios/Podfile
use_react_native!(
  :new_arch_enabled => true
)

// 3. 迁移原生模块为 TurboModules
// 旧模块 -> TurboModule + Codegen Spec

// 4. 迁移原生组件为 Fabric
// 旧 ViewManager -> Fabric Component + Codegen
```

```javascript
// 迁移兼容层 - 同时支持新旧架构
// modules/NativeMyModule.ts
import { TurboModuleRegistry, NativeModules } from 'react-native';

const isTurboModuleEnabled = global.__turboModuleProxy != null;

const MyModule = isTurboModuleEnabled
  ? TurboModuleRegistry.get('MyModule')
  : NativeModules.MyModule;

export default MyModule;
```

## 四、新旧架构对比

| 特性 | 旧架构 | 新架构 |
|------|--------|--------|
| 通信方式 | Bridge (异步) | JSI (同步) |
| 渲染器 | Paper (Stack) | Fabric (Tree) |
| 模块加载 | 启动时全部加载 | 按需加载 (TurboModules) |
| 布局引擎 | Yoga (异步) | Yoga (同步) |
| 并发支持 | 不支持 | 支持 Concurrent Features |
| 类型安全 | 无 | Codegen 自动生成 |

## 五、注意事项与常见陷阱

1. **第三方库兼容性**：迁移前检查所有依赖是否支持新架构
2. **原生模块迁移**：旧的原生模块需要重新实现为 TurboModules
3. **渐进式迁移**：可以在项目级别启用新架构，个别模块保持旧实现
4. **Hermes 调试**：Hermes 调试协议与 Chrome V8 不同，需要使用专用工具
5. **性能测试**：迁移后进行性能基准测试，确保优化效果
