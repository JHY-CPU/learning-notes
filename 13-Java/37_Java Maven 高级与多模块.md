# Java Maven 高级与多模块


## 🏗️ Maven 高级与多模块


多模块项目设计、BOM 依赖管理、Nexus 私有仓库、Maven Release 发布流程、Maven Enforcer 插件、最佳实践。


## 多模块项目


```
// ========== 多模块 Maven 项目 ==========
// 大型项目拆分为多个模块, 共享父 POM 配置
// 模块间可相互依赖

// ========== 项目结构 ==========
my-project/
├── pom.xml                  // 父 POM (packaging=pom)
├── my-common/               // 公共模块 (工具类、DTO)
│   └── pom.xml
├── my-core/                 // 核心业务模块
│   └── pom.xml
├── my-service/              // 服务层模块
│   └── pom.xml
├── my-web/                  // Web 接口模块
│   └── pom.xml
└── my-boot/                 // Spring Boot 启动模块
    └── pom.xml

// ========== 父 POM ==========
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-project</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <packaging>pom</packaging>  <!-- 父模块必须是 pom -->

    <!-- 聚合子模块 -->
    <modules>
        <module>my-common</module>
        <module>my-core</module>
        <module>my-service</module>
        <module>my-web</module>
        <module>my-boot</module>
    </modules>

    <!-- 全局属性 -->
    <properties>
        <java.version>17</java.version>
        <spring-boot.version>3.2.0</spring-boot.version>
        <spring-cloud.version>2023.0.0</spring-cloud.version>
        <mapstruct.version>1.5.5.Final</mapstruct.version>
    </properties>

    <!-- 统一依赖管理 (子模块无需 version) -->
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-dependencies</artifactId>
                <version>${spring-boot.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
            <dependency>
                <groupId>org.mapstruct</groupId>
                <artifactId>mapstruct</artifactId>
                <version>${mapstruct.version}</version>
            </dependency>
            <!-- 内部模块 -->
            <dependency>
                <groupId>com.example</groupId>
                <artifactId>my-common</artifactId>
                <version>${project.version}</version>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <!-- 所有子模块共用的依赖 -->
    <dependencies>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <scope>provided</scope>
        </dependency>
    </dependencies>
</project>

// ========== 子模块 pom.xml ==========
<project>
    <parent>
        <groupId>com.example</groupId>
        <artifactId>my-project</artifactId>
        <version>1.0.0-SNAPSHOT</version>
        <relativePath>../pom.xml</relativePath>
    </parent>

    <artifactId>my-core</artifactId>
    <!-- 继承父 POM 的 groupId/version -->

    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>my-common</artifactId>
            <!-- version 从父 POM dependencyManagement 继承 -->
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
            <!-- version 从 spring-boot-dependencies BOM 继承 -->
        </dependency>
    </dependencies>
</project>
```


## BOM 依赖管理


```
// ========== BOM (Bill of Materials) ==========
// 专门用于管理依赖版本的神 POM
// 让使用者无需指定版本, 统一管理

// ========== 自定义 BOM ==========
// my-bom/pom.xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-bom</artifactId>
    <version>1.0.0</version>
    <packaging>pom</packaging>

    <properties>
        <guava.version>33.0.0-jre</guava.version>
        <lombok.version>1.18.30</lombok.version>
        <mapstruct.version>1.5.5.Final</mapstruct.version>
    </properties>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>com.google.guava</groupId>
                <artifactId>guava</artifactId>
                <version>${guava.version}</version>
            </dependency>
            <dependency>
                <groupId>org.projectlombok</groupId>
                <artifactId>lombok</artifactId>
                <version>${lombok.version}</version>
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>

// ========== 使用 BOM ==========
// 项目引入 BOM
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>my-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>  <!-- import 作用域 -->
        </dependency>
    </dependencies>
</dependencyManagement>

// 使用 BOM 中的依赖, 无需指定版本:
<dependencies>
    <dependency>
        <groupId>com.google.guava</groupId>
        <artifactId>guava</artifactId>
        <!-- version 由 BOM 提供 -->
    </dependency>
</dependencies>

// ========== 常用 BOM ==========
// spring-boot-dependencies     — Spring Boot 所有依赖
// spring-cloud-dependencies    — Spring Cloud
// jackson-bom                  — Jackson
// junit-bom                    — JUnit 5
// testcontainers-bom           — Testcontainers

// ========== BOM 的好处 ==========
// 1. 统一版本管理
// 2. 避免依赖冲突
// 3. 升级只需改 BOM 版本
// 4. 跨项目共享版本标准
```


## Nexus 私有仓库


```
// ========== Nexus Repository Manager ==========
// 私有 Maven 仓库: 存储内部制品 + 代理中央仓库

// ========== 仓库类型 ==========
// hosted   — 托管仓库 (存内部制品)
// proxy    — 代理仓库 (缓存中央仓库)
// group    — 组合仓库 (合并多个仓库)

// 常见配置:
// maven-releases    — 正式版 (hosted, 不可覆盖)
// maven-snapshots   — 快照版 (hosted, 允许覆盖)
// maven-central     — 中央仓库代理 (proxy)
// maven-public      — group 组合以上所有

// ========== 配置私服地址 ==========
// pom.xml 配置分发
<distributionManagement>
    <repository>
        <id>nexus-releases</id>
        <name>Nexus Release Repository</name>
        <url>https://nexus.example.com/repository/maven-releases/</url>
    </repository>
    <snapshotRepository>
        <id>nexus-snapshots</id>
        <name>Nexus Snapshot Repository</name>
        <url>https://nexus.example.com/repository/maven-snapshots/</url>
    </snapshotRepository>
</distributionManagement>

// ========== settings.xml 配置认证 ==========
<servers>
    <server>
        <id>nexus-releases</id>
        <username>deploy-user</username>
        <password>${env.NEXUS_PASSWORD}</password>  <!-- 环境变量! -->
    </server>
    <server>
        <id>nexus-snapshots</id>
        <username>deploy-user</username>
        <password>${env.NEXUS_PASSWORD}</password>
    </server>
</servers>

// ========== 配置镜像代理 ==========
// settings.xml 中配置阿里云镜像
<mirrors>
    <mirror>
        <id>nexus-public</id>
        <mirrorOf>*</mirrorOf>    <!-- 拦截所有仓库 -->
        <url>https://nexus.example.com/repository/maven-public/</url>
    </mirror>
</mirrors>

// ========== 发布到私服 ==========
mvn deploy                      // 发布到 distributionManagement 配置的地址
mvn deploy -P prod              // 使用特定 profile 发布

// ========== 从私服下载 ==========
// 所有依赖从私服获取 (私服再从中央仓库同步)
// 内网开发无需访问外网
```


## Maven Release 与 Enforcer


```
// ========== Maven Release 流程 ==========
// 用于正式版本发布

// 1. 安装 maven-release-plugin (默认已内置)
// 2. 准备发布:
mvn release:prepare
//    - 检查无未提交的修改
//    - 移除 -SNAPSHOT (1.0.0-SNAPSHOT → 1.0.0)
//    - 更新 POM 版本
//    - 创建 tag (git tag my-project-1.0.0)
//    - 推进到下一个 SNAPSHOT (1.0.1-SNAPSHOT)

// 3. 执行发布:
mvn release:perform
//    - 从 tag 签出代码
//    - 执行 mvn deploy
//    - 发布到 Nexus 私服

// 4. 回滚:
mvn release:rollback

// ========== release 配置 ==========
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-release-plugin</artifactId>
    <version>3.0.1</version>
    <configuration>
        <tagNameFormat>v{project.version}</tagNameFormat>
        <autoVersionSubmodules>true</autoVersionSubmodules>
        <pushChanges>false</pushChanges>
    </configuration>
</plugin>

// ========== Maven Enforcer ==========
// 强制执行构建规则
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-enforcer-plugin</artifactId>
    <version>3.4.1</version>
    <executions>
        <execution>
            <id>enforce-rules</id>
            <goals><goal>enforce</goal></goals>
            <configuration>
                <rules>
                    <!-- 最低 Maven 版本 -->
                    <requireMavenVersion>
                        <version>3.8.0</version>
                    </requireMavenVersion>
                    <!-- 最低 JDK 版本 -->
                    <requireJavaVersion>
                        <version>17</version>
                    </requireJavaVersion>
                    <!--禁止依赖冲突 -->
                    <dependencyConvergence/>
                    <!-- 禁止 SNAPSHOT 依赖 -->
                    <requireReleaseDeps>
                        <message>正式发布不能依赖 SNAPSHOT 版本!</message>
                    </requireReleaseDeps>
                    <!-- 环境属性 -->
                    <requireEnvironmentVariable>
                        <variableName>NEXUS_PASSWORD</variableName>
                    </requireEnvironmentVariable>
                </rules>
            </configuration>
        </execution>
    </executions>
</plugin>

// ========== 最佳实践 ==========
// 1. 统一 parent POM 管理所有版本
// 2. 使用 ${project.version} 引用内部模块版本
// 3. settings.xml 不要提交到 Git (含密码)
// 4. 使用环境变量注入敏感信息
// 5. SNAPSHOT 用于开发, release 用于发布
// 6. 使用 enforcer 强制执行规范
// 7. 多模块时使用 -pl 指定模块: mvn -pl my-core compile
// 8. -am 同时构建依赖: mvn -pl my-web -am package
```


> **Note:** 💡 Maven 高级要点: 多模块父 POM packaging=pom + modules 聚合; dependencyManagement 统一版本, 子模块自动继承; BOM 通过 import scope 管理依赖版本; Nexus 私服 hosted/proxy/group; mvn deploy 发布; mvn release:prepare + release:perform 正式发布; enforcer 强制执行规则; 最佳实践: 统一版本管理、环境变量加密、SNAPSHOT 开发/Release 发布。


## 练习


<!-- Converted from: 37_Java Maven 高级与多模块.html -->
