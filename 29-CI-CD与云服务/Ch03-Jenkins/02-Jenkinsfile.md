# Jenkinsfile

## 一、概念说明

Jenkinsfile是用Groovy语法定义的流水线脚本，存储在代码仓库中，实现Pipeline as Code。

## 二、声明式流水线

```groovy
pipeline {
    agent any
    
    environment {
        NODE_ENV = 'production'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/repo.git'
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh './deploy.sh'
            }
        }
    }
    
    post {
        success {
            echo 'Build successful!'
        }
        failure {
            echo 'Build failed!'
        }
    }
}
```

## 三、脚本式流水线

```groovy
node {
    stage('Checkout') {
        git 'https://github.com/repo.git'
    }
    
    stage('Build') {
        sh 'npm install'
    }
    
    stage('Test') {
        sh 'npm test'
    }
}
```

## 四、注意事项

1. **语法选择**：推荐声明式流水线
2. **版本控制**：Jenkinsfile提交到Git
3. **错误处理**：使用post块处理结果
4. **环境变量**：使用environment块定义

## 五、完整声明式流水线

```groovy
pipeline {
    agent {
        docker {
            image 'node:20-alpine'
            args '-v $HOME/.npm:/root/.npm'
        }
    }

    environment {
        CI = 'true'
        NODE_ENV = 'production'
        DOCKER_CREDS = credentials('docker-hub')
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git describe --tags || echo "no-tag"'
            }
        }

        stage('Install') {
            steps {
                sh 'npm ci'
            }
        }

        stage('Parallel Checks') {
            parallel {
                stage('Lint') {
                    steps {
                        sh 'npm run lint'
                    }
                }
                stage('Type Check') {
                    steps {
                        sh 'npm run type-check'
                    }
                }
            }
        }

        stage('Test') {
            steps {
                sh 'npm test -- --coverage'
            }
            post {
                always {
                    junit 'test-results/junit.xml'
                    publishHTML(target: [
                        reportDir: 'coverage',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }

        stage('Build') {
            steps {
                sh 'npm run build'
                stash includes: 'dist/**', name: 'dist'
            }
        }

        stage('Deploy Staging') {
            when {
                branch 'develop'
            }
            steps {
                unstash 'dist'
                sh './deploy.sh staging'
            }
        }

        stage('Deploy Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                unstash 'dist'
                sh './deploy.sh production'
            }
        }
    }

    post {
        success {
            slackSend(color: 'good', message: "Build SUCCESS: ${env.JOB_NAME} #${env.BUILD_NUMBER}")
        }
        failure {
            slackSend(color: 'danger', message: "Build FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER}")
        }
        always {
            cleanWs()
        }
    }
}
```

## 六、共享库集成

```groovy
// vars/myPipeline.groovy (共享库)
def call(Map config) {
    pipeline {
        agent any
        stages {
            stage('Build') {
                steps {
                    script {
                        if (config.language == 'node') {
                            sh 'npm ci && npm run build'
                        } else if (config.language == 'python') {
                            sh 'pip install -r requirements.txt'
                        }
                    }
                }
            }
            stage('Test') {
                steps {
                    sh config.testCommand ?: 'npm test'
                }
            }
            stage('Deploy') {
                when {
                    branch 'main'
                }
                steps {
                    sh "./deploy.sh ${config.environment}"
                }
            }
        }
    }
}
```

```groovy
// Jenkinsfile (使用共享库)
@Library('my-shared-lib') _

myPipeline(
    language: 'node',
    testCommand: 'npm test -- --coverage',
    environment: 'staging'
)
```

## 七、多分支Pipeline配置

```groovy
// Jenkinsfile for Multibranch Pipeline
pipeline {
    agent any

    triggers {
        pollSCM('H/5 * * * *')  // 每5分钟检查变更
    }

    stages {
        stage('PR Check') {
            when {
                changeRequest()
            }
            steps {
                echo "Building PR: ${env.CHANGE_ID}"
                sh 'npm ci && npm test'
            }
        }

        stage('Main Build') {
            when {
                branch 'main'
            }
            steps {
                sh 'npm ci && npm run build && npm test'
            }
        }

        stage('Release') {
            when {
                buildingTag()
            }
            steps {
                echo "Building tag: ${env.TAG_NAME}"
                sh 'npm run build:release'
            }
        }
    }
}
```
