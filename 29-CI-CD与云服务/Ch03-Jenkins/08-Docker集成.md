# Docker集成

## 一、Docker Agent

```groovy
pipeline {
    agent {
        docker {
            image 'node:20-alpine'
            args '-v $HOME/.npm:/root/.npm'
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
    }
}
```

## 二、构建Docker镜像

```groovy
stage('Build Image') {
    steps {
        script {
            def image = docker.build("myapp:${env.BUILD_NUMBER}")
            image.push()
            image.push('latest')
        }
    }
}
```

## 三、Docker Compose

```groovy
stage('Integration Test') {
    steps {
        sh 'docker-compose up -d'
        sh 'npm run test:integration'
        sh 'docker-compose down'
    }
}
```

## 四、注意事项

1. **镜像拉取**：配置Docker Registry凭据
2. **清理容器**：构建后清理Docker容器
3. **缓存利用**：利用Docker层缓存
4. **安全扫描**：集成Docker镜像扫描
