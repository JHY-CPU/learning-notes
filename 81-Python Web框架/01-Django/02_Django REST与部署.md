# Django REST 与部署


## Django REST Framework 与部署


DRFDockerGunicorn


Django REST Framework（DRF）是 Django 的 REST API 扩展，提供序列化器、视图集、认证、权限、分页等 API 开发必备功能。


## DRF 序列化器（Serializers）


```
ModelSerializer：
  from rest_framework import serializers
  from .models import Post, Comment

  class CommentSerializer(serializers.ModelSerializer):
      author_name = serializers.CharField(source='author.username', read_only=True)

      class Meta:
          model = Comment
          fields = ['id', 'content', 'author', 'author_name', 'created_at']
          read_only_fields = ['author', 'created_at']

  class PostSerializer(serializers.ModelSerializer):
      author_name = serializers.CharField(source='author.username', read_only=True)
      comments = CommentSerializer(many=True, read_only=True)
      comment_count = serializers.IntegerField(source='comments.count', read_only=True)
      tags = serializers.PrimaryKeyRelatedField(many=True, queryset=Tag.objects.all())

      class Meta:
          model = Post
          fields = [
              'id', 'title', 'content', 'author', 'author_name',
              'status', 'tags', 'comments', 'comment_count',
              'created_at', 'updated_at',
          ]
          read_only_fields = ['author', 'created_at', 'updated_at']

      def validate_title(self, value):
          if len(value) < 5:
              raise serializers.ValidationError('标题至少5个字符')
          return value

      def validate(self, data):
          if data.get('status') == 'published' and not data.get('content'):
              raise serializers.ValidationError('发布状态必须有内容')
          return data

      def create(self, validated_data):
          tags = validated_data.pop('tags', [])
          post = Post.objects.create(**validated_data)
          post.tags.set(tags)
          return post

嵌套序列化与写入：
  # 嵌套创建
  class PostWithCommentsSerializer(serializers.ModelSerializer):
      comments = CommentSerializer(many=True)

      def create(self, validated_data):
          comments_data = validated_data.pop('comments')
          post = Post.objects.create(**validated_data)
          for comment_data in comments_data:
              Comment.objects.create(post=post, **comment_data)
          return post

字段选择（Sparse Fieldsets）：
  class DynamicFieldsSerializer(serializers.ModelSerializer):
      class Meta:
          model = Post
          fields = '__all__'

      def __init__(self, *args, **kwargs):
          fields = kwargs.pop('fields', None)
          super().__init__(*args, **kwargs)
          if fields:
              allowed = set(fields)
              existing = set(self.fields)
              for field_name in existing - allowed:
                  self.fields.pop(field_name)

  # 使用：只返回指定字段
  # GET /api/posts/?fields=id,title,status
```


## DRF 视图集与路由


```
ViewSet 与 Router：
  from rest_framework import viewsets, permissions, filters
  from rest_framework.decorators import action
  from rest_framework.response import Response
  from django_filters.rest_framework import DjangoFilterBackend

  class PostViewSet(viewsets.ModelViewSet):
      queryset = Post.objects.all()
      serializer_class = PostSerializer
      permission_classes = [permissions.IsAuthenticatedOrReadOnly]
      filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
      filterset_fields = ['status', 'author']
      search_fields = ['title', 'content']
      ordering_fields = ['created_at', 'view_count']
      pagination_class = PageNumberPagination

      def perform_create(self, serializer):
          serializer.save(author=self.request.user)

      def get_queryset(self):
          if self.action == 'list':
              return Post.objects.select_related('author').prefetch_related('tags')
          return super().get_queryset()

      # 自定义 action
      @action(detail=True, methods=['post'])
      def publish(self, request, pk=None):
          post = self.get_object()
          post.status = 'published'
          post.save()
          return Response({'status': 'published'})

      @action(detail=False, methods=['get'])
      def my_posts(self, request):
          posts = Post.objects.filter(author=request.user)
          serializer = self.get_serializer(posts, many=True)
          return Response(serializer.data)

路由配置：
  from django.urls import path, include
  from rest_framework.routers import DefaultRouter
  from .views import PostViewSet, CommentViewSet

  router = DefaultRouter()
  router.register('posts', PostViewSet)
  router.register('comments', CommentViewSet)

  urlpatterns = [
      path('api/', include(router.urls)),
  ]

  # 自动生成的路由：
  # GET    /api/posts/          → list
  # POST   /api/posts/          → create
  # GET    /api/posts/{id}/     → retrieve
  # PUT    /api/posts/{id}/     → update
  # PATCH  /api/posts/{id}/     → partial_update
  # DELETE /api/posts/{id}/     → destroy
  # POST   /api/posts/{id}/publish/  → custom action
  # GET    /api/posts/my_posts/      → custom action

函数视图（Function-Based Views）：
  from rest_framework.decorators import api_view
  from rest_framework import status

  @api_view(['GET', 'POST'])
  def post_list(request):
      if request.method == 'GET':
          posts = Post.objects.all()
          serializer = PostSerializer(posts, many=True)
          return Response(serializer.data)

      elif request.method == 'POST':
          serializer = PostSerializer(data=request.data)
          if serializer.is_valid():
              serializer.save(author=request.user)
              return Response(serializer.data, status=status.HTTP_201_CREATED)
          return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```


## 认证、权限与分页


```
JWT 认证（djangorestframework-simplejwt）：
  # settings.py
  REST_FRAMEWORK = {
      'DEFAULT_AUTHENTICATION_CLASSES': [
          'rest_framework_simplejwt.authentication.JWTAuthentication',
      ],
  }

  from datetime import timedelta
  SIMPLE_JWT = {
      'ACCESS_TOKEN_LIFETIME': timedelta(minutes=15),
      'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
      'ROTATE_REFRESH_TOKENS': True,
  }

  # urls.py
  from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
  urlpatterns = [
      path('api/token/', TokenObtainPairView.as_view()),
      path('api/token/refresh/', TokenRefreshView.as_view()),
  ]

  # 使用
  # POST /api/token/ {"username": "xxx", "password": "xxx"}
  # → {"access": "xxx", "refresh": "xxx"}
  # 请求时：Authorization: Bearer xxx

权限类：
  from rest_framework import permissions

  class IsAuthorOrReadOnly(permissions.BasePermission):
      def has_object_permission(self, request, view, obj):
          if request.method in permissions.SAFE_METHODS:
              return True
          return obj.author == request.user

  # 使用
  class PostViewSet(viewsets.ModelViewSet):
      permission_classes = [permissions.IsAuthenticated, IsAuthorOrReadOnly]

  # 内置权限
  # IsAuthenticated - 必须登录
  # IsAdminUser - 必须管理员
  # AllowAny - 允许所有
  # DjangoModelPermissions - 基于模型权限

分页：
  from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination, CursorPagination

  # 页码分页
  class StandardPagination(PageNumberPagination):
      page_size = 20
      page_size_query_param = 'page_size'
      max_page_size = 100

  # 偏移分页
  class StandardPagination(LimitOffsetPagination):
      default_limit = 20
      max_limit = 100

  # 游标分页（推荐大数据量）
  class StandardPagination(CursorPagination):
      page_size = 20
      ordering = '-created_at'

  # 全局设置
  REST_FRAMEWORK = {
      'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
      'PAGE_SIZE': 20,
  }

  # 响应格式
  {
      "count": 150,
      "next": "http://api.example.com/posts/?page=2",
      "previous": null,
      "results": [...]
  }

限流：
  REST_FRAMEWORK = {
      'DEFAULT_THROTTLE_CLASSES': [
          'rest_framework.throttling.AnonRateThrottle',
          'rest_framework.throttling.UserRateThrottle',
      ],
      'DEFAULT_THROTTLE_RATES': {
          'anon': '100/hour',
          'user': '1000/hour',
      },
  }
```


## Docker 与生产部署


```
Dockerfile：
  FROM python:3.11-slim

  WORKDIR /app

  # 安装依赖
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt gunicorn

  # 复制项目
  COPY . .

  # 收集静态文件
  RUN python manage.py collectstatic --noinput

  # 运行
  EXPOSE 8000
  CMD ["gunicorn", "myproject.wsgi:application", \
       "--bind", "0.0.0.0:8000", \
       "--workers", "4", \
       "--threads", "2", \
       "--timeout", "120"]

docker-compose.yml：
  version: '3.8'

  services:
    web:
      build: .
      command: >
        gunicorn myproject.wsgi:application
        --bind 0.0.0.0:8000
        --workers 4
        --timeout 120
      volumes:
        - static_volume:/app/static
        - media_volume:/app/media
      expose:
        - "8000"
      depends_on:
        - db
        - redis
      environment:
        - DEBUG=0
        - DATABASE_URL=postgres://user:pass@db:5432/mydb
        - REDIS_URL=redis://redis:6379/0

    nginx:
      image: nginx:latest
      ports:
        - "80:80"
        - "443:443"
      volumes:
        - ./nginx.conf:/etc/nginx/conf.d/default.conf
        - static_volume:/app/static
        - media_volume:/app/media
      depends_on:
        - web

    db:
      image: postgres:15
      volumes:
        - postgres_data:/var/lib/postgresql/data
      environment:
        - POSTGRES_DB=mydb
        - POSTGRES_USER=user
        - POSTGRES_PASSWORD=pass

    redis:
      image: redis:7-alpine

  volumes:
    static_volume:
    media_volume:
    postgres_data:

Nginx 配置（nginx.conf）：
  upstream django {
      server web:8000;
  }

  server {
      listen 80;
      server_name www.example.com;

      location /static/ {
          alias /app/static/;
          expires 30d;
      }

      location /media/ {
          alias /app/media/;
      }

      location / {
          proxy_pass http://django;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }
  }

Gunicorn 配置（gunicorn.conf.py）：
  bind = "0.0.0.0:8000"
  workers = 4               # 2 × CPU核心数 + 1
  threads = 2               # 每个 worker 的线程数
  worker_class = "gthread"  # 线程模式
  timeout = 120
  keepalive = 5
  max_requests = 1000       # worker 处理请求数后重启（防内存泄漏）
  max_requests_jitter = 50  # 随机抖动，防止同时重启
  accesslog = "-"
  errorlog = "-"

部署命令：
  # 构建和启动
  docker-compose up -d --build

  # 执行迁移
  docker-compose exec web python manage.py migrate

  # 创建超级用户
  docker-compose exec web python manage.py createsuperuser

  # 查看日志
  docker-compose logs -f web
```


> **Note:** DRF 提供了完整的 REST API 开发工具链：ModelSerializer 序列化、ViewSet 视图集、Router 自动路由、JWT 认证、分页、限流等。生产部署推荐 Docker Compose（Gunicorn + Nginx + PostgreSQL + Redis）架构。Gunicorn worker 数量建议 2×CPU+1，配合 max_requests 防内存泄漏。


<!-- Converted from: 02_Django REST与部署.html -->
