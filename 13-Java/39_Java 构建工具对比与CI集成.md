# Java 构建工具对比与 CI 集成


## ⚖️ 构建工具对比与 CI 集成


Maven vs Gradle 深度对比、选型指南、GitHub Actions/GitLab CI 集成 Maven/Gradle、构建速度优化、常见问题排查。


## Maven vs Gradle 深度对比


```
// ========== Maven vs Gradle 全面对比 ==========

// ┌───────────────────┬──────────────────────┬───────────────────────┐
// │ 维度              │ Maven                │ Gradle                │
// ├───────────────────┼──────────────────────┼───────────────────────┤
// │ 配置文件          │ XML (pom.xml)        │ Groovy / Kotlin DSL   │
// │                   │ 声明式, 冗长         │ 编程式, 简洁          │
// │ 构建模型          │ 固定生命周期          │ 任务 DAG              │
// │                   │ (phase → goal)       │ (灵活可编程)          │
// │ 性能              │ 较慢                 │ 快 2-10 倍            │
// │ 增量编译          │ 无内置               │ 内置增量编译          │
// │ 守护进程          │ 无 (每次启动 JVM)     │ 有 (长期运行)         │
// │ 构建缓存          │ 无内置               │ 内置 (跨项目)         │
// │ 并行构建          │ -T 参数              │ 默认并行              │
// │ 依赖配置粒度      │ 5 种 scope           │ implementation/api    │
// │                   │                      │ + compileOnly 等      │
// │ 依赖传递          │ 自动传递              │ 自动传递              │
// │                   │ 最短路径优先          │ 默认最新              │
// │ Gradle Wrapper    │ 无 (mvnw 可选)       │ 官方推荐              │
// │ 版本目录          │ dependencyManagement │ Version Catalog       │
// │ 自定义逻辑        │ 需写插件 (Java)      │ 直接在 DSL 中写      │
// │ 学习曲线          │ 较低                 │ 较高                  │
// │ Android 支持      │ 不适用               │ 官方工具              │
// │ 市场份额          │ 传统企业为主          │ 新项目增长最快        │
// │ 生态成熟度        │ 最成熟               │ 快速发展              │
// └───────────────────┴──────────────────────┴───────────────────────┘

// ========== 选型决策 ==========
// 选 Maven 当:
//   - 团队熟悉 XML / 企业标准化
//   - 已有大量 Maven 项目
//   - 需要严格的生命周期规范
//   - 项目简单, 性能不是瓶颈

// 选 Gradle 当:
//   - 构建速度要求高 (大型项目)
//   - Android 项目 (强制)
//   - 需要灵活构建逻辑
//   - 微服务 / 多模块项目
//   - 新项目 (推荐 Gradle)

// 实际趋势:
// Spring Boot 官方同时支持 Maven + Gradle
// 但 Gradle 已成为 JVM 生态构建工具的事实标准
```


## GitHub Actions CI 集成


```
// ========== GitHub Actions + Maven ==========
// .github/workflows/maven-ci.yml

name: Java CI with Maven

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up JDK 17
      uses: actions/setup-java@v4
      with:
        java-version: '17'
        distribution: 'temurin'
        cache: maven                    # 缓存 ~/.m2/repository

    - name: Build & Test
      run: mvn clean verify -B         # -B: batch mode (非交互)

    - name: Upload Artifact
      uses: actions/upload-artifact@v3
      with:
        name: app-jar
        path: target/*.jar

// ========== GitHub Actions + Gradle ==========
name: Java CI with Gradle

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up JDK 17
      uses: actions/setup-java@v4
      with:
        java-version: '17'
        distribution: 'temurin'
        cache: gradle                   # 缓存 ~/.gradle/caches

    - name: Setup Gradle
      uses: gradle/actions/setup-gradle@v3

    - name: Build with Gradle
      run: ./gradlew build              # 使用 Wrapper!

    - name: Upload Test Report
      if: always()                      # 即使测试失败也上传
      uses: actions/upload-artifact@v3
      with:
        name: test-report
        path: build/reports/tests/

// ========== 缓存策略 ==========
// Maven: 缓存 ~/.m2/repository
// Gradle: 缓存 ~/.gradle/caches + ~/.gradle/wrapper
// 能显著减少 CI 构建时间 (从 5 分钟 → 1 分钟)

// ========== 矩阵构建 ==========
// 多 JDK 版本测试
jobs:
  build:
    strategy:
      matrix:
        java: [17, 21]
        os: [ubuntu-latest, windows-latest]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-java@v4
      with:
        java-version: ${{ matrix.java }}
        distribution: 'temurin'
    - run: mvn clean verify
```


## GitLab CI / Jenkins 集成


```
// ========== GitLab CI ==========
// .gitlab-ci.yml

image: eclipse-temurin:17-jdk

variables:
  MAVEN_OPTS: "-Dmaven.repo.local=$CI_PROJECT_DIR/.m2/repository"

cache:
  paths:
    - .m2/repository/                  # 缓存 Maven 依赖
    - .gradle/                         # 缓存 Gradle 依赖

stages:
  - build
  - test
  - package

maven-build:
  stage: build
  script:
    - mvn compile
  artifacts:
    paths:
      - target/classes/

maven-test:
  stage: test
  script:
    - mvn test
  artifacts:
    reports:
      junit: target/surefire-reports/TEST-*.xml

maven-package:
  stage: package
  script:
    - mvn package -DskipTests
  artifacts:
    paths:
      - target/*.jar

// ========== Jenkins Declarative Pipeline ==========
// Jenkinsfile

pipeline {
    agent any

    tools {
        maven 'Maven-3.9'              // Jenkins 全局工具配置
        jdk 'JDK-17'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                sh 'mvn clean compile'
            }
        }

        stage('Test') {
            steps {
                sh 'mvn test'
            }
            post {
                always {
                    junit 'target/surefire-reports/*.xml'
                }
            }
        }

        stage('Package') {
            steps {
                sh 'mvn package -DskipTests'
            }
        }

        stage('Deploy to Nexus') {
            steps {
                sh 'mvn deploy'
            }
        }
    }

    post {
        failure {
            emailext(
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                to: 'team@example.com',
                body: "Check build: ${env.BUILD_URL}"
            )
        }
    }
}
```


## 构建优化与问题排查


```
// ========== Maven 构建优化 ==========

// 1. 多线程构建
mvn -T 4 clean install                // 4 线程
mvn -T 1C clean install               // CPU 核心数

// 2. 跳过不必要的阶段
mvn package -DskipTests               // 跳过测试
mvn verify -Dmaven.test.skip=true     // 完全跳过

// 3. 离线模式 (依赖已下载)
mvn -o clean install

// 4. 增量编译 (Maven 3.8+)
// 使用 maven-compiler-plugin 增量特性
mvn compile -Dmaven.compiler.incremental=true

// 5. 使用最新的 Maven 版本 (3.9+ 性能改进)

// 6. settings.xml 配置国内镜像
// 阿里云镜像显著提高依赖下载速度

// ========== Gradle 构建优化 ==========

// 1. 确保开启守护进程 (默认开启)
// gradle.properties:
org.gradle.daemon=true

// 2. 并行构建
org.gradle.parallel=true

// 3. 构建缓存
org.gradle.caching=true

// 4. 配置按需加载
org.gradle.configureondemand=true

// 5. 使用 Gradle Wrapper 最新版本
gradle wrapper --gradle-version 8.5

// 6. 增量编译 (默认开启)
tasks.withType(JavaCompile) {
    options.incremental = true
}

// 7. 文件系统监控 (Gradle 8+)
// ./gradlew build --watch

// ========== 常见问题排查 ==========

// 1. 依赖冲突
// Maven:  mvn dependency:tree
// Gradle: ./gradlew dependencies
// 查看冲突, 用 exclusion 排除

// 2. 编译缓存问题
// Maven:  删除 ~/.m2/repository 中的相关 jar
// Gradle: ./gradlew cleanBuildCache

// 3. 测试内存不足
// Maven surefire 配置:
//   <forkCount>2</forkCount>
//   <argLine>-Xmx1024m</argLine>

// Gradle: 在 build.gradle 中
// tasks.withType(Test) {
//     maxHeapSize = '1G'
// }

// 4. CI 构建慢
// - 启用依赖缓存
// - 使用国内镜像
// - 并行构建
// - 区分单元测试和集成测试
//   (surefire 单元测试, failsafe 集成测试)

// 5. "Could not find artifact"
// - 检查仓库配置
// - 检查依赖坐标是否正确
// - 检查本地仓库是否已损坏 (删除重试)

// 6. Gradle Daemon 内存
// gradle.properties:
// org.gradle.jvmargs=-Xmx2048m -XX:MaxMetaspaceSize=512m
```


> **Note:** 💡 CI/CD 要点: Maven 和 Gradle 均支持 CI 集成; GitHub Actions setup-java + cache 加速构建; GitLab CI artifacts 传递构建产物; Jenkins pipeline 声明式语法; 构建优化: 多线程/缓存/镜像/offline; 依赖冲突用 dependency:tree / dependencies 排查; Gradle 守护进程+并行+缓存 = 最快构建。


## 练习


<!-- Converted from: 39_Java 构建工具对比与CI集成.html -->
