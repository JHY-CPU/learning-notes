# Java Maven 生命周期与插件


## 🔄 Maven 生命周期与插件


3 大生命周期 (default/clean/site)、阶段绑定、插件目标 (goal)、常用插件配置、Profile 多环境构建。


## Maven 三大生命周期


```
// ========== Maven 生命周期 ==========
// Maven 有 3 个独立生命周期, 各有多个阶段 (phase)
// 阶段按顺序执行, 执行靠后的会自动执行之前所有阶段

// ========== 1. default (项目部署) ==========
// 最常用, 处理项目构建和部署
validate          → 验证项目正确性, 必要信息可用
initialize        → 初始化构建状态 (创建目录)
generate-sources  → 生成源代码
process-sources   → 处理源代码 (过滤/替换)
generate-resources → 生成资源文件
process-resources → 复制资源到目标目录
compile           → 编译源代码
process-classes   → 处理编译后的文件 (增强/字节码)
generate-test-sources  → 生成测试源代码
process-test-sources   → 处理测试源代码
generate-test-resources → 生成测试资源
process-test-resources  → 复制测试资源到目标目录
test-compile       → 编译测试代码
process-test-classes   → 处理测试编译后文件
test               → 运行测试
prepare-package    → 打包前准备
package            → 打包 (jar/war)
pre-integration-test  → 集成测试前准备
integration-test   → 集成测试
post-integration-test → 集成测试后清理
verify             → 验证包有效
install            → 安装到本地仓库
deploy             → 部署到远程仓库

// mvn test → 执行 validate → ... → test
// mvn package → validate → ... → package
// mvn install → validate → ... → install

// ========== 2. clean (清理) ==========
pre-clean    → 清理前
clean        → 删除 target/ 目录
post-clean   → 清理后

// ========== 3. site (文档生成) ==========
pre-site     → 文档生成前
site         → 生成项目文档
post-site    → 文档生成后
site-deploy  → 部署文档到服务器
```


## 插件与目标


```
// ========== 插件 (Plugin) 与目标 (Goal) ==========
// 生命周期阶段 → 绑定到 → 插件目标
// 每个插件包含多个目标 (goal)

// ========== 内置插件绑定 ==========
// ┌──────────────┬──────────────────────────────────────┐
// │ 生命周期阶段  │ 插件:目标                           │
// ├──────────────┼──────────────────────────────────────┤
// │ process-res  │ maven-resources-plugin:resources     │
// │ compile      │ maven-compiler-plugin:compile        │
// │ test-compile │ maven-compiler-plugin:testCompile    │
// │ test         │ maven-surefire-plugin:test           │
// │ package      │ maven-jar-plugin:jar / war plugin   │
// │ install      │ maven-install-plugin:install         │
// │ deploy       │ maven-deploy-plugin:deploy           │
// │ clean        │ maven-clean-plugin:clean             │
// │ site         │ maven-site-plugin:site               │
// └──────────────┴──────────────────────────────────────┘

// ========== 常用插件配置 ==========
<build>
    <plugins>
        <!-- 编译插件 -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.11.0</version>
            <configuration>
                <source>17</source>
                <target>17</target>
                <encoding>UTF-8</encoding>
                <parameters>true</parameters>  <!-- 保留参数名 -->
                <!-- --parameters 让反射可获取参数名 -->
            </configuration>
        </plugin>

        <!-- 测试插件 -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-surefire-plugin</artifactId>
            <version>3.1.2</version>
            <configuration>
                <includes>
                    <include>**/*Test.java</include>
                    <include>**/*Tests.java</include>
                </includes>
                <excludes>
                    <exclude>**/*IntegrationTest.java</exclude>
                </excludes>
                <forkCount>4</forkCount>
                <reuseForks>true</reuseForks>
            </configuration>
        </plugin>

        <!-- Jar 包插件 -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-jar-plugin</artifactId>
            <version>3.3.0</version>
            <configuration>
                <archive>
                    <manifest>
                        <mainClass>com.example.Main</mainClass>
                        <addClasspath>true</addClasspath>
                    </manifest>
                </archive>
            </configuration>
        </plugin>
    </plugins>
</build>
```


## Fat Jar 与 Assembly


```
// ========== 打包可执行 JAR ==========

// ========== maven-shade-plugin (打 fat jar) ==========
// 将所有依赖合并到一个 jar 中
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-shade-plugin</artifactId>
    <version>3.5.1</version>
    <executions>
        <execution>
            <phase>package</phase>
            <goals><goal>shade</goal></goals>
            <configuration>
                <transformers>
                    <transformer implementation=
                        "org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                        <mainClass>com.example.Main</mainClass>
                    </transformer>
                    <!-- 合并 META-INF/services -->
                    <transformer implementation=
                        "org.apache.maven.plugins.shade.resource.ServicesResourceTransformer"/>
                </transformers>
                <!-- 排除不需要的文件 -->
                <filters>
                    <filter>
                        <artifact>*:*</artifact>
                        <excludes>
                            <exclude>META-INF/*.SF</exclude>
                            <exclude>META-INF/*.DSA</exclude>
                            <exclude>META-INF/*.RSA</exclude>
                        </excludes>
                    </filter>
                </filters>
            </configuration>
        </execution>
    </executions>
</plugin>

// ========== maven-assembly-plugin ==========
// 更灵活的打包, 可自定义格式
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-assembly-plugin</artifactId>
    <version>3.6.0</version>
    <configuration>
        <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
        </descriptorRefs>
        <archive>
            <manifest>
                <mainClass>com.example.Main</mainClass>
            </manifest>
        </archive>
    </configuration>
    <executions>
        <execution>
            <id>make-assembly</id>
            <phase>package</phase>
            <goals><goal>single</goal></goals>
        </execution>
    </executions>
</plugin>

// ========== Spring Boot Maven Plugin ==========
<plugin>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-maven-plugin</artifactId>
    <version>3.2.0</version>
    <configuration>
        <excludes>
            <exclude>
                <groupId>org.projectlombok</groupId>
                <artifactId>lombok</artifactId>
            </exclude>
        </excludes>
    </configuration>
</plugin>
```


## Profile 与环境


```
// ========== Maven Profile ==========
// 根据不同环境使用不同配置
// 常见: dev / test / prod

// ========== 定义 Profile ==========
<profiles>
    <!-- 开发环境 -->
    <profile>
        <id>dev</id>
        <activation>
            <activeByDefault>true</activeByDefault>
        </activation>
        <properties>
            <env>dev</env>
            <db.url>jdbc:h2:mem:testdb</db.url>
            <logging.level>DEBUG</logging.level>
        </properties>
    </profile>

    <!-- 生产环境 -->
    <profile>
        <id>prod</id>
        <properties>
            <env>prod</env>
            <db.url>jdbc:mysql://prod-server:3306/mydb</db.url>
            <logging.level>WARN</logging.level>
        </properties>
    </profile>

    <!-- 按条件激活 -->
    <profile>
        <id>java17</id>
        <activation>
            <jdk>17</jdk>                        <!-- JDK 版本 -->
            <os>
                <name>Windows 11</name>            <!-- OS 条件 -->
            </os>
            <property>
                <name>environment</name>
                <value>test</value>               <!-- -Denvironment=test -->
            </property>
        </activation>
        <dependencies>
            <dependency>
                <groupId>com.h2database</groupId>
                <artifactId>h2</artifactId>
                <scope>test</scope>
            </dependency>
        </dependencies>
    </profile>
</profiles>

// ========== Profile 激活方式 ==========
// 1. 默认激活: <activeByDefault>true</activeByDefault>
// 2. 命令行:   mvn package -Pprod
// 3. 系统属性: mvn package -Denvironment=test
// 4. 文件存在: <activation><file><exists>...
// 5. JDK/OS:  <jdk>17</jdk>

// ========== 资源过滤 ==========
// 让 profile 属性替换资源文件中的 ${...}
<build>
    <resources>
        <resource>
            <directory>src/main/resources</directory>
            <filtering>true</filtering>  <!-- 启用变量替换 -->
            <includes>
                <include>application-*.yml</include>
                <include>*.properties</include>
            </includes>
        </resource>
    </resources>
</build>

// application-dev.yml 中使用: ${db.url}
// mvn package -Pdev → 替换为 jdbc:h2:mem:testdb

// ========== settings.xml 安全配置 ==========
// 服务器密码加密存储
<servers>
    <server>
        <id>private-repo</id>
        <username>deploy-user</username>
        <password>{encrypted-password}</password>
    </server>
</servers>
```


## 实用技巧


```
// ========== Maven 实用技巧 ==========

// ========== 跳过测试 ==========
mvn package -DskipTests           // 编译测试但不运行
mvn package -Dmaven.test.skip=true  // 完全不编译测试

// ========== 多线程构建 ==========
mvn -T 4 clean install            // 4 线程并行
mvn -T 1C clean install           // CPU 核心数线程

// ========== 离线模式 ==========
mvn -o clean install              // 只使用本地仓库

// ========== 调试输出 ==========
mvn -X clean install              // 调试日志
mvn -e clean install              // 错误详情

// ========== 检查更新 ==========
mvn versions:display-dependency-updates   // 检查依赖更新
mvn versions:display-plugin-updates       // 检查插件更新
mvn versions:display-property-updates     // 检查属性更新

// ========== 生成项目 ==========
mvn archetype:generate           // 交互式创建项目
// Maven 原型 (archetype) 模板:
// maven-archetype-quickstart      // 简单 Java 项目
// maven-archetype-webapp          // Web 应用

// ========== 传递依赖解决 ==========
// 查看为何引入某个依赖:
mvn dependency:tree -Dincludes=com.google.guava
// 输出:
// com.example:my-app:jar:1.0
// └─ org.springframework.boot:spring-boot-starter-web:jar:3.2.0
//    └─ com.fasterxml.jackson.core:jackson-databind:jar:2.15.0
//       └─ com.fasterxml.jackson.core:jackson-core:jar:2.15.0

// ========== 分析依赖 ==========
mvn dependency:analyze
// 报告:
// [WARNING] Unused declared dependencies: ...
// [WARNING] Used undeclared dependencies: ...

// ========== 生成 Javadoc ==========
mvn javadoc:javadoc               // 生成 Javadoc
mvn site                          // 生成项目站点文档

// ========== Maven Wrapper ==========
// 类似 Gradle Wrapper, 锁定 Maven 版本
mvn -N wrapper:wrapper -Dmaven=3.9.6
// 生成 mvnw / mvnw.cmd
```


> **Note:** 💡 生命周期与插件要点: 3 大生命周期 default/clean/site; 阶段按顺序执行, 靠后包含之前; 插件目标绑定到生命周期阶段; maven-compiler-plugin 编译配置; maven-surefire-plugin 测试; maven-shade-plugin fat jar; Profile 多环境 (dev/prod/test); -P 激活 profile; -T 多线程构建; -o 离线模式; dependency:tree 分析依赖树。


## 练习


<!-- Converted from: 36_Java Maven 生命周期与插件.html -->
