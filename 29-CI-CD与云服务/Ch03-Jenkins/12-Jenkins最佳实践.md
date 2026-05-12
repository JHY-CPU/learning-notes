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

## 五、共享库最佳实践

```groovy
// vars/buildNode.groovy
def call(Map config = [:]) {
    def nodeVersion = config.nodeVersion ?: '20'
    def testCommand = config.testCommand ?: 'npm test'

    pipeline {
        agent {
            docker {
                image "node:${nodeVersion}-alpine"
                args '-v $HOME/.npm:/root/.npm'
            }
        }

        options {
            timeout(time: 20, unit: 'MINUTES')
            disableConcurrentBuilds()
            timestamps()
        }

        stages {
            stage('Install') {
                steps {
                    sh 'npm ci'
                }
            }

            stage('Lint') {
                steps {
                    sh 'npm run lint'
                }
            }

            stage('Test') {
                steps {
                    sh testCommand
                }
                post {
                    always {
                        junit allowEmptyResults: true,
                              testResults: 'test-results/*.xml'
                    }
                }
            }

            stage('Build') {
                steps {
                    sh 'npm run build'
                }
            }
        }

        post {
            always {
                cleanWs()
            }
            failure {
                script {
                    if (env.SLACK_WEBHOOK) {
                        sh """
                            curl -X POST -H 'Content-type: application/json' \
                            --data '{"text":"Build FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER}"}' \
                            ${env.SLACK_WEBHOOK}
                        """
                    }
                }
            }
        }
    }
}
```

## 六、Job DSL批量管理

```groovy
// jobs.groovy - 使用Job DSL批量创建任务
folder('microservices') {
    description('微服务构建任务')
}

['user-service', 'order-service', 'payment-service'].each { service ->
    pipelineJob("microservices/${service}") {
        definition {
            cpsScm {
                scm {
                    git {
                        remote {
                            url("https://github.com/myorg/${service}.git")
                            credentials('github-token')
                        }
                        branches('main', 'develop')
                    }
                }
                scriptPath('Jenkinsfile')
            }
        }
        triggers {
            githubPush()
        }
        parameters {
            stringParam('VERSION', '1.0.0', '版本号')
            choiceParam('ENV', ['dev', 'staging', 'prod'], '部署环境')
        }
    }
}
```

## 七、备份策略

```bash
# 定期备份JENKINS_HOME
# /etc/cron.d/jenkins-backup
0 2 * * * root /opt/scripts/backup-jenkins.sh

# backup-jenkins.sh
#!/bin/bash
BACKUP_DIR="/backup/jenkins"
DATE=$(date +%Y%m%d_%H%M%S)
JENKINS_HOME="/var/jenkins_home"

# 停止Jenkins
systemctl stop jenkins

# 创建备份
tar czf "${BACKUP_DIR}/jenkins_${DATE}.tar.gz" \
  --exclude="${JENKINS_HOME}/workspace" \
  --exclude="${JENKINS_HOME}/builds/*/archive" \
  "${JENKINS_HOME}"

# 启动Jenkins
systemctl start jenkins

# 清理30天前的备份
find "${BACKUP_DIR}" -name "jenkins_*.tar.gz" -mtime +30 -delete
```

## 八、安全加固清单

```groovy
// jenkins.yaml (JCasC安全配置)
jenkins:
  securityRealm:
    local:
      allowsSignup: false
  authorizationStrategy:
    loggedInUsersCanDoAnything:
      allowsAnonymousRead: false
  remotingSecurity:
    enabled: true

security:
  queueItemAuthenticator:
    authenticators:
      - global:
          strategy: triggeringUsersAuthorizationStrategy

# 启用CSRF保护
# Manage Jenkins > Configure Global Security
# - Enable proxy compatibility: 关闭
# - Prevent Cross Site Request Forgery exploits: 开启
```
