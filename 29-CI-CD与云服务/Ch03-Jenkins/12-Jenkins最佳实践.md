# Jenkins最佳实践

## 一、流水线最佳实践

```groovy
// 1. 使用声明式流水线
pipeline {
    agent any
    
    // 2. 使用环境变量
    environment {
        APP_NAME = 'myapp'
    }
    
    // 3. 参数化构建
    parameters {
        string(name: 'VERSION')
    }
    
    stages {
        // 4. 阶段命名清晰
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build & Test') {
            parallel {
                stage('Build') { steps { sh 'make' } }
                stage('Test') { steps { sh 'make test' } }
            }
        }
    }
    
    // 5. 后置处理
    post {
        always { cleanWs() }
        failure { slackSend message: 'Build failed!' }
    }
}
```

## 二、安全最佳实践

```bash
# 1. 配置认证和授权
# 2. 使用凭据管理敏感信息
# 3. 定期更新Jenkins和插件
# 4. 限制脚本执行权限
# 5. 启用CSRF保护
```

## 三、性能优化

```bash
# 1. 使用分布式构建
# 2. 清理旧构建
# 3. 优化磁盘空间
# 4. 配置构建保留策略
# 5. 使用构建缓存
```

## 四、维护检查

- [ ] 定期备份JENKINS_HOME
- [ ] 安全更新Jenkins和插件
- [ ] 监控磁盘空间
- [ ] 清理旧构建记录
- [ ] 检查Agent节点状态
- [ ] 审计构建日志
