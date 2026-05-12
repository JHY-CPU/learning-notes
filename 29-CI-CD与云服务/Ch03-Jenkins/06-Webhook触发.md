# Webhook触发

## 一、概念说明

Webhook允许代码仓库（GitHub/GitLab）在代码推送时自动触发Jenkins构建。

## 二、配置GitHub Webhook

```bash
# GitHub设置
# Settings > Webhooks > Add webhook
# Payload URL: http://jenkins-url/github-webhook/
# Content type: application/json
# Events: Just the push event

# Jenkins配置
# Pipeline > Build Triggers
# GitHub hook trigger for GITScm polling
```

## 三、配置GitLab Webhook

```bash
# GitLab设置
# Settings > Webhooks
# URL: http://jenkins-url/project/job-name
# Trigger: Push events

# Jenkins配置
# Pipeline > Build Triggers
# Build when a change is pushed to GitLab
```

## 四、Generic Webhook

```groovy
pipeline {
    triggers {
        GenericTrigger(
            genericVariables: [
                [key: 'ref', value: '$.ref']
            ],
            causeString: 'Triggered by webhook',
            token: 'my-token'
        )
    }
    stages {
        stage('Build') {
            steps {
                echo "Branch: ${env.ref}"
            }
        }
    }
}
```

## 五、注意事项

1. **内网访问**：确保GitHub/GitLab能访问Jenkins
2. **安全验证**：验证Webhook来源
3. **日志检查**：检查Webhook触发日志
4. **重试机制**：配置Webhook重试

## 六、GitHub Webhook完整配置

```groovy
// Jenkinsfile - GitHub Webhook触发
pipeline {
    agent any
    triggers {
        GenericTrigger(
            genericVariables: [
                [key: 'ref', value: '$.ref'],
                [key: 'action', value: '$.action'],
                [key: 'repository', value: '$.repository.full_name']
            ],
            causeString: 'Triggered by GitHub push to ${ref}',
            token: 'my-webhook-token',
            tokenCredentialId: '',
            printPostContent: true,
            printContributedVariables: true,
            silentResponse: false,
            regexpFilterText: '$ref',
            regexpFilterExpression: '^refs/heads/(main|develop)$'
        )
    }
    stages {
        stage('Build') {
            steps {
                echo "Branch: ${ref}"
                echo "Action: ${action}"
                echo "Repo: ${repository}"
            }
        }
    }
}
```

## 七、Webhook安全验证

```groovy
// 验证GitHub Webhook签名
pipeline {
    agent any
    triggers {
        GenericTrigger(
            genericVariables: [
                [key: 'payload', value: '$'],
                [key: 'signature', value: '$.headers.X-Hub-Signature-256']
            ],
            token: 'my-secret-token',
            causeString: 'GitHub webhook'
        )
    }
    stages {
        stage('Verify') {
            steps {
                script {
                    // 验证签名
                    def expectedSig = sh(
                        script: "echo -n '${payload}' | openssl dgst -sha256 -hmac '${WEBHOOK_SECRET}' | cut -d' ' -f2",
                        returnStdout: true
                    ).trim()
                    if (!signature.endsWith(expectedSig)) {
                        error('Invalid webhook signature')
                    }
                }
            }
        }
    }
}
```

## 八、Ngrok内网穿透（开发环境）

```bash
# 安装ngrok
npm install -g ngrok

# 启动隧道
ngrok http 8080

# 获取公网URL
# https://abc123.ngrok.io -> http://localhost:8080

# 在GitHub中配置Webhook
# Payload URL: https://abc123.ngrok.io/github-webhook/
# Content type: application/json
# Secret: your-webhook-secret
```

## 九、Webhook调试技巧

```bash
# 查看Jenkins Webhook日志
# Manage Jenkins > System Log > Add "org.jenkinsci.plugins"

# 使用curl模拟Webhook
curl -X POST http://jenkins-url/github-webhook/ \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: push" \
  -d '{
    "ref": "refs/heads/main",
    "repository": {"full_name": "owner/repo"},
    "pusher": {"name": "developer"}
  }'

# 检查Webhook是否触发
curl -u admin:token http://jenkins-url/job/myjob/lastBuild/api/json | jq '.actions[].causes'
```
