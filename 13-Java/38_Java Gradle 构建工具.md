# Java Gradle 构建工具


## 📦 Gradle 构建工具


Gradle 基础 (Groovy/Kotlin DSL)、构建脚本、任务依赖与排序、插件机制、依赖管理、多项目构建、Gradle Wrapper、性能优化。


## Gradle 基础


```
// ========== Gradle ==========
// 基于 DAG (有向无环图) 的构建工具
// 核心概念: Project + Task
// 每个 build.gradle 是一个 Project
// 构建工作由 Task 组成的 DAG 完成

// ========== Groovy DSL 语法 ==========
// build.gradle

plugins {
    id 'java'                          // 应用 Java 插件
    id 'application'                   // 应用 Application 插件
}

group = 'com.example'
version = '1.0.0'
sourceCompatibility = '17'

repositories {
    mavenCentral()                     // 中央仓库
    mavenLocal()                       // 本地仓库 (~/.m2/repository)
    maven { url 'https://repo.spring.io/milestone' }
}

dependencies {
    // 配置名称 = scope
    implementation 'org.springframework.boot:spring-boot-starter-web:3.2.0'
    compileOnly 'org.projectlombok:lombok:1.18.30'
    annotationProcessor 'org.projectlombok:lombok:1.18.30'
    runtimeOnly 'com.h2database:h2:2.2.224'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
}

application {
    mainClass = 'com.example.App'
}

tasks.named('test') {
    useJUnitPlatform()
}

// ========== Kotlin DSL 语法 ==========
// build.gradle.kts

// plugins {
//     java
//     id("org.springframework.boot") version "3.2.0"
// }
//
// group = "com.example"
// version = "1.0.0"
//
// repositories {
//     mavenCentral()
// }
//
// dependencies {
//     implementation("org.springframework.boot:spring-boot-starter-web")
// }
//
// tasks.test {
//     useJUnitPlatform()
// }

// ========== settings.gradle ==========
// 项目配置入口
rootProject.name = 'my-app'
include 'common'
include 'service'
include 'web'

// settings.gradle.kts
// rootProject.name = "my-app"
// include("common", "service", "web")

// ========== gradle.properties ==========
// 项目级属性
org.gradle.jvmargs=-Xmx2g -XX:MaxMetaspaceSize=512m
org.gradle.parallel=true
org.gradle.caching=true
org.gradle.daemon=true
```


## Task 体系


```
// ========== Gradle Task ==========
// Gradle 的核心: 一切工作都是 Task
// Task 构成 DAG (有向无环图)

// ========== 定义 Task ==========
tasks.register('hello') {
    doLast {
        println 'Hello, Gradle!'
    }
}

tasks.register('copyFile', Copy) {
    from 'src/config'
    into 'build/config'
    include '*.yml'
}

// ========== 任务依赖 ==========
tasks.register('compile') {
    doLast { println 'Compiling...' }
}

tasks.register('test') {
    dependsOn 'compile'              // test 依赖 compile
    doLast { println 'Testing...' }
}

tasks.register('package') {
    dependsOn 'test'                 // package 依赖 test
    doLast { println 'Packaging...' }
}

// 执行 gradle package → 自动先执行 compile → test → package

// ========== 任务排序 ==========
tasks.register('taskA') {
    mustRunAfter 'taskB'             // A 必须在 B 之后
    shouldRunAfter 'taskB'           // 建议顺序 (弱约束)
}

tasks.register('taskFinal') {
    finalizedBy 'cleanup'            // 无论成功失败都执行 cleanup
}

// ========== 任务类型 ==========
// Gradle 内置任务类型:
// Copy      — 文件复制
// Delete    — 删除文件
// Exec      — 执行外部命令
// JavaExec  — 运行 Java 主类
// Jar       — 创建 JAR 包
// Zip/Tar   — 压缩打包
// Test      — 运行测试

tasks.register('runApp', JavaExec) {
    mainClass = 'com.example.App'
    classpath = sourceSets.main.runtimeClasspath
    jvmArgs '-Xmx512m'
}

tasks.register('printVersion') {
    doLast {
        println "Version: ${project.version}"
    }
}

// ========== 任务增量构建 ==========
// 定义输入输出, 无变化则跳过
tasks.register('processData') {
    inputs.file('input.txt')
    outputs.file('output.txt')
    doLast {
        // 只有 input.txt 变化时才执行
    }
}

// ========== 默认任务 ==========
defaultTasks 'clean', 'build'        // 执行 gradle 默认运行的任务
```


## 依赖管理


```
// ========== Gradle 依赖配置 ==========
// Gradle 有更细粒度的依赖配置 (取代 Maven scope)

// ========== Java 插件配置名称 ==========
// ┌─────────────────┬──────────────────────────────────────┐
// │ 配置名          │ 用途                                 │
// ├─────────────────┼──────────────────────────────────────┤
// │ implementation  │ 编译+运行 (不暴露给消费者)           │
// │ api             │ 编译+运行 (暴露给消费者, 用于库)     │
// │ compileOnly     │ 仅编译 (类似 Maven provided)         │
// │ runtimeOnly     │ 仅运行 (类似 Maven runtime)          │
// │ testImplementation │ 测试编译+运行                     │
// │ testCompileOnly    │ 仅测试编译                        │
// │ testRuntimeOnly    │ 仅测试运行                        │
// │ annotationProcessor │ 注解处理器 (Lombok/MapStruct)   │
// └─────────────────┴──────────────────────────────────────┘

// ========== implementation vs api ==========
// implementation: 依赖不会传递到消费者的编译路径
//   模块 A implementation B, B implementation C
//   A 不能使用 C 的类 (编译隔离)
//   优点: 编译更快, 避免依赖泄漏

// api: 依赖会传递到消费者的编译路径
//   模块 A api B, B api C
//   A 可以使用 C 的类
//   用于库项目暴露 API 依赖

// ========== 依赖声明方式 ==========
// 完整坐标字符串
implementation 'org.springframework.boot:spring-boot-starter-web:3.2.0'

// Map 方式
implementation group: 'org.springframework.boot',
               name: 'spring-boot-starter-web',
               version: '3.2.0'

// 文件依赖
implementation files('libs/local-lib.jar')
implementation fileTree(dir: 'libs', include: '*.jar')

// 项目依赖 (多模块)
implementation project(':common')
implementation project(':service')

// ========== 依赖约束 ==========
// 强制版本
implementation('org.apache.httpcomponents:httpclient:4.5.14') {
    force = true                     // 强制使用此版本
    exclude group: 'commons-logging' // 排除传递依赖
    transitive = false               // 禁用传递依赖
}

// ========== 版本目录 (Version Catalog) ==========
// libs.versions.toml (gradle/ 目录下)
// [versions]
// spring-boot = "3.2.0"
// lombok = "1.18.30"
//
// [libraries]
// spring-boot-starter-web = { module = "org.springframework.boot:spring-boot-starter-web", version.ref = "spring-boot" }
// lombok = { module = "org.projectlombok:lombok", version.ref = "lombok" }

// build.gradle.kts
// dependencies {
//     implementation(libs.spring.boot.starter.web)
//     compileOnly(libs.lombok)
// }
```


## Gradle Wrapper 与多项目


```
// ========== Gradle Wrapper ==========
// 版本锁定 + 无需安装 Gradle
// 推荐所有项目使用!

// 生成 Wrapper:
gradle wrapper --gradle-version 8.5

// 文件结构:
my-project/
├── gradlew              // Unix 执行脚本
├── gradlew.bat          // Windows 执行脚本
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties  // 版本配置

// 使用:
./gradlew build           // Linux/Mac
gradlew build             // Windows
// 自动下载对应版本 Gradle, 无需全局安装

// ========== 多项目构建 ==========
// settings.gradle
rootProject.name = 'my-platform'
include 'common'
include 'core'
include 'service:user-service'
include 'service:order-service'
include 'web:admin-web'
include 'web:api-web'

// 根项目 build.gradle
subprojects {
    apply plugin: 'java'

    repositories {
        mavenCentral()
    }

    dependencies {
        testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
    }
}

// 所有子项目共享 Java 插件 + 测试依赖

// ========== 子项目专用配置 ==========
// service/user-service/build.gradle
dependencies {
    implementation project(':common')    // 依赖 common 模块
    implementation 'org.springframework.boot:spring-boot-starter-web'
}

// ========== 构建性能优化 ==========
// gradle.properties
org.gradle.jvmargs=-Xmx2048m -XX:MaxMetaspaceSize=512m
org.gradle.parallel=true              // 并行构建
org.gradle.caching=true               // 构建缓存
org.gradle.daemon=true                // 守护进程 (默认开启)
org.gradle.configureondemand=true     // 仅配置必要的项目

// 构建扫描:
// ./gradlew build --scan              // 生成构建报告

// 增量编译:
// 默认已启用, 只重新编译修改的文件

// 构建缓存:
// 跨项目复用构建输出
// ./gradlew build --build-cache
```


## Gradle 插件


```
// ========== Gradle 插件 ==========
// 插件扩展 Gradle 功能
// 2 类: 脚本插件 + 二进制插件

// ========== 常见插件 ==========
// Java 相关:
// java                        — Java 编译+测试+打包
// java-library                — Java 库 (用 api 替代 implementation)
// application                 — 可运行应用
// groovy / scala              — 其他 JVM 语言

// Framework:
// org.springframework.boot    — Spring Boot 打包+运行
// io.spring.dependency-management — Spring 依赖管理
// com.google.protobuf         — Protobuf 编译

// 代码质量:
// checkstyle                  — 代码风格检查
// pmd                         — 静态分析
// spotbugs                    — Bug 检测
// jacoco                      — 测试覆盖率

// ========== Spring Boot 插件 ==========
plugins {
    id 'java'
    id 'org.springframework.boot' version '3.2.0'
    id 'io.spring.dependency-management' version '1.1.4'
}

// spring-boot 插件提供:
// bootJar   — 打 fat jar (默认替代 jar task)
// bootRun   — 运行 Spring Boot 应用
// bootBuildImage — 构建 Docker 镜像 (使用 Buildpacks)

// bootBuildImage 示例:
// ./gradlew bootBuildImage
// 生成镜像: docker.io/library/my-app:1.0.0

// ========== 自定义插件 ==========
// buildSrc 目录共享插件逻辑
// buildSrc/src/main/groovy/my-conventions.gradle
//
// plugins {
//     id 'java'
// }
//
// java {
//     toolchain {
//         languageVersion = JavaLanguageVersion.of(17)
//     }
// }

// 使用: 其他项目 apply plugin: 'my-conventions'

// ========== 常用命令 ==========
// ./gradlew build              // 编译+测试+打包
// ./gradlew clean              // 清理 build/
// ./gradlew test               // 运行测试
// ./gradlew check              // 质量检查 (含 test/lint)
// ./gradlew bootRun            // Spring Boot 启动
// ./gradlew tasks              // 列出所有 task
// ./gradlew dependencies       // 依赖树
// ./gradlew dependencyUpdates  // 检查依赖更新
// ./gradlew build --scan       // 构建扫描报告
// ./gradlew --watch            // 文件变化自动重构建 (Gradle 8+)
// ./gradlew build -x test      // 跳过测试

// ========== Maven vs Gradle 命令对照 ==========
// Maven                  Gradle
// mvn clean             → ./gradlew clean
// mvn compile           → ./gradlew compileJava
// mvn test              → ./gradlew test
// mvn package           → ./gradlew build
// mvn install           → ./gradlew publishToMavenLocal
// mvn deploy            → ./gradlew publish
// mvn dependency:tree   → ./gradlew dependencies
```


> **Note:** 💡 Gradle 要点: Groovy/Kotlin DSL 两种语法; Task DAG 定义构建流程; implementation/api 细粒度依赖管理; Wrapper 版本锁定无需安装; 多项目 subprojects 共享配置; 守护进程 + 并行 + 缓存 = 快速构建; plugins 扩展功能; bootBuildImage 构建 Docker 镜像; 比 Maven 快 2-10 倍。


## 练习


<!-- Converted from: 38_Java Gradle 构建工具.html -->
