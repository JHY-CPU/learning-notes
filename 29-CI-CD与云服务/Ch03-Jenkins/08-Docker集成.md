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

## 四、最佳实践

1. **Registry凭据**：在Jenkins Credentials中配置Docker Registry用户名密码，使用`withCredentials`安全引用
2. **层缓存优化**：Dockerfile中将不常变化的指令（如`COPY package*.json`）放在前面，充分利用层缓存
3. **多阶段构建**：使用multi-stage build减小最终镜像体积，只保留运行时必需文件
4. **镜像扫描**：集成Trivy或Snyk进行安全扫描，发现CVE漏洞及时修复

## 五、常见陷阱

1. **僵尸容器**：构建失败时容器未被清理，长期积累占用磁盘空间，建议使用`post { always { sh 'docker-compose down' } }`
2. **权限问题**：Jenkins用户不在docker组中导致无权操作Docker，需要将jenkins用户加入docker组
3. **镜像拉取慢**：未配置镜像加速器导致拉取超时，配置阿里云或DaoCloud镜像源
4. **数据卷冲突**：多个Job同时使用同一数据卷可能导致数据竞争，使用`${BUILD_NUMBER}`隔离

## 六、完整Docker Pipeline

```groovy
pipeline {
    agent {
        docker {
            image 'node:20-alpine'
            args '-v $HOME/.npm:/root/.npm --network=host'
        }
    }

    stages {
        stage('Build App') {
            steps {
                sh 'npm ci'
                sh 'npm run build'
            }
        }

        stage('Docker Build & Push') {
            agent any
            steps {
                script {
                    docker.withRegistry('https://registry.example.com', 'registry-creds') {
                        def appImage = docker.build("myapp:${env.BUILD_NUMBER}")
                        appImage.push()
                        appImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy') {
            agent any
            steps {
                sshagent(['deploy-key']) {
                    sh '''
                        ssh deploy@server << 'EOF'
                            docker pull registry.example.com/myapp:latest
                            docker-compose -f /opt/myapp/docker-compose.yml up -d
                        EOF
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker system prune -f'
        }
    }
}
```

## 七、Docker多架构构建

```groovy
stage('Multi-arch Build') {
    steps {
        script {
            sh '''
                docker buildx create --use --name multiarch
                docker buildx build \
                    --platform linux/amd64,linux/arm64 \
                    -t registry.example.com/myapp:${BUILD_NUMBER} \
                    --push \
                    .
            '''
        }
    }
}
```

## 八、Docker Registry认证管理

```groovy
pipeline {
    agent any
    stages {
        stage('Push to Registry') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'docker-registry',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin registry.example.com
                        docker build -t registry.example.com/myapp:${BUILD_NUMBER} .
                        docker push registry.example.com/myapp:${BUILD_NUMBER}
                        docker logout registry.example.com
                    '''
                }
            }
        }
    }
}
```

## 九、Docker Compose集成测试

```groovy
stage('Integration Test') {
    steps {
        script {
            // 启动测试环境
            sh 'docker-compose -f docker-compose.test.yml up -d'
            // 等待服务就绪
            sh '''
                for i in $(seq 1 30); do
                    if docker-compose -f docker-compose.test.yml exec -T db pg_isready -U postgres; then
                        break
                    fi
                    sleep 2
                done
            '''
            // 运行测试
            sh 'docker-compose -f docker-compose.test.yml run --rm app npm test'
        }
    }
    post {
        always {
            sh 'docker-compose -f docker-compose.test.yml down -v --remove-orphans'
        }
    }
}
```

## 十、Docker构建监控

```groovy
stage('Build Stats') {
    steps {
        script {
            def imageSize = sh(
                script: "docker image inspect myapp:${BUILD_NUMBER} --format='{{.Size}}'",
                returnStdout: true
            ).trim()
            def imageSizeMB = (imageSize.toLong() / 1024 / 1024).round(2)
            echo "Image size: ${imageSizeMB} MB"

            // 检查镜像大小限制
            if (imageSizeMB > 500) {
                error("Image size (${imageSizeMB}MB) exceeds 500MB limit")
            }

            // 查看构建层
            sh "docker history myapp:${BUILD_NUMBER}"
        }
    }
}
```
