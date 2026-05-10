# Java Maven 构建工具


## 🏗️ Java Maven 构建工具


Maven pom.xml（坐标/依赖/仓库/插件/生命周期）、Maven 命令（clean/compile/test/package/install）、Gradle 基础、Maven vs Gradle 对比。


## Maven 基础与 pom.xml


```
// ========== Maven (Java 构建工具) ==========
// 项目构建: 编译、测试、打包、部署
// 依赖管理: 自动下载 jar 包
// 约定优于配置

// ========== pom.xml 基本结构 ==========
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <!-- ========== GAV 坐标 (唯一标识) ========== -->
    <groupId>com.example</groupId>          <!-- 组织/公司域名反写 -->
    <artifactId>my-app</artifactId>         <!-- 项目名 -->
    <version>1.0.0-SNAPSHOT</version>      <!-- 版本 (SNAPSHOT=开发版) -->
    <packaging>jar</packaging>              <!-- jar/war/pom -->

    <!-- ========== 属性 ========== -->
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <!-- ========== 依赖 ========== -->
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-api</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>              <!-- 作用域: compile/test/provided/runtime/system -->
        </dependency>

        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.4.14</version>
        </dependency>
    </dependencies>

    <!-- ========== 插件 ========== -->
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```


## Maven 依赖机制


```
// ========== Maven 依赖机制 ==========

// ========== 依赖传递 ==========
// A → B → C: A 自动依赖 C
// mvn dependency:tree  查看依赖树
// mvn dependency:analyze  检查未使用/未声明依赖

// ========== scope (作用域) ==========
// ┌──────────┬──────────────────────────────────┐
// │ scope    │ 用途                             │
// ├──────────┼──────────────────────────────────┤
// │ compile  │ 默认, 所有阶段可用               │
// │ provided │ 编译时需要, 运行时由 JDK/容器提供 │
// │          │ 例: servlet-api, Lombok          │
// │ runtime  │ 编译不需要, 运行需要             │
// │          │ 例: JDBC 驱动                    │
// │ test     │ 仅测试阶段                       │
// │          │ 例: JUnit, Mockito               │
// │ system   │ 类似 provided, 但需指定 path     │
// └──────────┴──────────────────────────────────┘

// ========== 排除传递依赖 ==========
<dependency>
    <groupId>com.example</groupId>
    <artifactId>lib-a</artifactId>
    <version>1.0</version>
    <exclusions>
        <exclusion>
            <groupId>com.example</groupId>
            <artifactId>lib-b</artifactId>  <!-- 排除 lib-b -->
        </exclusion>
    </exclusions>
</dependency>

// ========== 依赖冲突 ==========
// 最短路径优先: A→B→C→X 1.0  vs  A→D→X 2.0  → 用 2.0
// 第一声明优先: 路径相同则先声明者获胜
// 强制版本: 在 <dependencyManagement> 中指定

// ========== dependencyManagement ==========
// 在父 POM 统一管理版本, 子模块不需指定 version
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>com.google.guava</groupId>
            <artifactId>guava</artifactId>
            <version>33.0.0-jre</version>
        </dependency>
    </dependencies>
</dependencyManagement>

// ========== 仓库 ==========
// 本地仓库: ~/.m2/repository
// 中央仓库: https://repo.maven.apache.org/maven2
// 私有仓库: Nexus / Artifactory

// 配置镜像 (settings.xml)
<mirrors>
    <mirror>
        <id>aliyun</id>
        <mirrorOf>central</mirrorOf>
        <name>阿里云公共仓库</name>
        <url>https://maven.aliyun.com/repository/public</url>
    </mirror>
</mirrors>
```


## Maven 生命周期与命令


```
// ========== Maven 生命周期 ==========
// 3 个独立生命周期, 阶段按顺序执行

// ========== 1. default (主要) ==========
// validate     → 验证项目正确性
// compile      → 编译源代码
// test         → 运行测试 (JUnit)
// package      → 打包 jar/war
// verify       → 集成测试检查
// install      → 安装到本地仓库 (~/.m2)
// deploy       → 部署到远程仓库

// 执行后置阶段会自动执行之前所有阶段:
// mvn package  → 自动执行 validate → compile → test → package

// ========== 2. clean ==========
// clean  → 删除 target/ 目录

// ========== 3. site ==========
// site  → 生成项目文档

// ========== 常用命令 ==========
// mvn clean                // 清理 target/
// mvn compile              // 编译
// mvn test                 // 运行测试
// mvn package              // 打包 jar/war
// mvn install              // 安装到本地仓库
// mvn deploy               // 部署到远程仓库
// mvn clean package        // 先清理再打包 (最常用)
// mvn clean install        // 先清理再安装
// mvn test -Dtest=MyTest   // 运行特定测试类
// mvn test -DskipTests     // 跳过测试 (编译阶段仍编译测试)
// mvn test -Dmaven.test.skip=true  // 完全跳过测试
// mvn dependency:tree      // 查看依赖树
// mvn dependency:analyze   // 分析依赖使用情况
// mvn help:effective-pom   // 显示最终 POM (含继承)
// mvn versions:display-dependency-updates  // 检查依赖更新

// ========== -SNAPSHOT 版本 ==========
// 1.0.0-SNAPSHOT = 开发中的不稳定版本
// mvn install 时自动从远程仓库拉取最新 SNAPSHOT
// 正式发布: mvn release:prepare + release:perform
```


## Maven 继承与聚合


```
// ========== 多模块项目 ==========

// ========== 父 POM (聚合 + 继承) ==========
// parent/pom.xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0.0</version>
    <packaging>pom</packaging>  <!-- 父模块必须是 pom -->

    <!-- 聚合: 子模块列表 -->
    <modules>
        <module>common</module>
        <module>service</module>
        <module>web</module>
    </modules>

    <!-- 统一管理版本 -->
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-dependencies</artifactId>
                <version>3.2.0</version>
                <type>pom</type>
                <scope>import</scope>  <!-- BOM 导入 -->
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>

// ========== 子模块 POM ==========
// common/pom.xml
<project>
    <parent>
        <groupId>com.example</groupId>
        <artifactId>parent</artifactId>
        <version>1.0.0</version>
        <relativePath>../pom.xml</relativePath>
    </parent>

    <artifactId>common</artifactId>
    <!-- 继承父 POM 的 version, 无需重复 -->

    <dependencies>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <!-- version 从父 POM 继承 -->
        </dependency>
    </dependencies>
</project>

// ========== BOM (Bill of Materials) ==========
// 统一管理一组依赖的版本
// 使用方导入:
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>my-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

// ========== 常用插件 ==========
// maven-compiler-plugin     — 编译配置 (source/target)
// maven-surefire-plugin     — 运行单元测试 (JUnit)
// maven-failsafe-plugin     — 运行集成测试
// maven-jar-plugin          — 打包 jar
// maven-war-plugin          — 打包 war
// maven-assembly-plugin     — 自定义打包 (含依赖)
// maven-shade-plugin        — 打 fat jar (含依赖)
// maven-source-plugin       — 打包源码
// maven-javadoc-plugin      — 生成 Javadoc
// spring-boot-maven-plugin  — Spring Boot 打包
```


## Gradle 基础


```
// ========== Gradle (Groovy/Kotlin DSL) ==========
// 基于 DAG (有向无环图) 的任务执行
// 增量构建: 只重新执行有变化的任务
// 构建缓存: 跨项目复用构建结果
// 更快的编译速度 (daemon 进程 + 增量编译)

// ========== build.gradle (Groovy DSL) ==========
plugins {
    id 'java'
    id 'org.springframework.boot' version '3.2.0'
    id 'io.spring.dependency-management' version '1.1.4'
}

group = 'com.example'
version = '1.0.0'
sourceCompatibility = '17'

repositories {
    mavenCentral()
    maven { url 'https://maven.aliyun.com/repository/public' }
}

dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    compileOnly 'org.projectlombok:lombok'
    annotationProcessor 'org.projectlombok:lombok'
    runtimeOnly 'com.h2database:h2'
    testImplementation 'org.springframework.boot:spring-boot-starter-test'
}

test {
    useJUnitPlatform()
}

// ========== build.gradle.kts (Kotlin DSL) ==========
plugins {
    java
    id("org.springframework.boot") version "3.2.0"
    id("io.spring.dependency-management") version "1.1.4"
}

group = "com.example"
version = "1.0.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    compileOnly("org.projectlombok:lombok")
    annotationProcessor("org.projectlombok:lombok")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

tasks.test {
    useJUnitPlatform()
}

// ========== Gradle 常用命令 ==========
// gradle build              // 编译+测试+打包
// gradle clean              // 清理 build/
// gradle test               // 运行测试
// gradle bootRun            // Spring Boot 启动
// gradle build --watch      // 持续构建
// gradle tasks              // 列出任务
// gradle dependencies       // 查看依赖树
// ./gradlew build           // Gradle Wrapper (推荐!)
```


## Maven vs Gradle


```
// ========== Maven vs Gradle 对比 ==========

// ┌──────────────┬──────────────────────────┬──────────────────────────┐
// │ 特性         │ Maven                    │ Gradle                   │
// ├──────────────┼──────────────────────────┼──────────────────────────┤
// │ 配置文件     │ XML (pom.xml)            │ Groovy/Kotlin DSL        │
// │              │ 声明式, 冗长             │ 简洁, 可编程             │
// │ 性能         │ 较慢 (无增量编译)        │ 快 (daemon+增量+缓存)    │
// │              │                          │ 比 Maven 快 2-10 倍      │
// │ 依赖管理     │ pom.xml 声明             │ 类似, 但支持动态版本     │
// │              │ 传递依赖+排除            │ 更灵活的条件依赖         │
// │ 构建逻辑     │ 插件+生命周期(固定)      │ 任务 DAG (灵活)          │
// │              │ 定制需插件               │ 可直接写逻辑             │
// │ 多模块       │ 父 POM + modules         │ 类似 settings.gradle     │
// │ 学习曲线     │ 较低 (XML 标准)          │ 较高 (需学 DSL)          │
// │ 生态         │ 最成熟                   │ 增长最快                 │
// │ 市场份额     │ 传统企业主导             │ Android/新项目首选        │
// │ 使用场景     │ 传统企业项目             │ Android/新项目/微服务     │
// │              │ Spring Boot 项目         │ Spring Boot 也支持        │
// └──────────────┴──────────────────────────┴──────────────────────────┘

// ========== 何时用 Maven ==========
// • 团队成员熟悉 XML
// • 传统企业项目
// • 需要严格标准化
// • 已有大量 Maven 项目

// ========== 何时用 Gradle ==========
// • Android 项目 (官方工具)
// • 需要快速构建
// • 需要灵活构建逻辑
// • 微服务/新项目
// • 多项目构建

// ========== settings.gradle (多模块) ==========
rootProject.name = 'my-project'
include 'common', 'service', 'web'

// ========== Gradle Wrapper ==========
// gradle wrapper --gradle-version 8.5
// 生成 gradlew + gradlew.bat
// 团队无需安装 Gradle, 用 gradlew 即可构建
// 版本锁定, 避免环境不一致
```


> **Note:** 💡 构建工具要点: Maven pom.xml GAV 坐标标识项目; 依赖自动传递 + scope 控制作用域; 生命周期 default/clean/site; 常用命令 clean/compile/test/package/install/deploy; dependencyManagement 统一版本; 多模块 parent+module; Gradle DSL 更简洁 + 性能更快; Gradle Wrapper 锁定版本; 新项目倾向 Gradle, 传统企业 Maven。


## 练习


<!-- Converted from: 35_Java Maven 构建工具.html -->
