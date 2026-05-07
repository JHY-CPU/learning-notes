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
