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
