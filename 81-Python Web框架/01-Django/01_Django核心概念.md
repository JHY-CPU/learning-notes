# Django 核心概念


## Django 核心概念


DjangoMTVORM


Django 是全栈 Web 框架，采用 MTV（Model-Template-View）架构模式，内置 ORM、Admin、认证等开箱即用功能，强调 DRY（Don't Repeat Yourself）原则。


## MTV 架构模式


```
MTV = Model + Template + View
  对比 MVC：
  ┌──────────┬──────────────────┐
  │  MVC     │  MTV             │
  ├──────────┼──────────────────┤
  │  Model   │  Model（数据层） │
  │  View    │  Template（展示）│
  │  Controller│ View（业务逻辑）│
  └──────────┴──────────────────┘

项目结构：
  myproject/
  ├── manage.py                # 管理命令
  ├── myproject/
  │   ├── settings.py          # 配置文件
  │   ├── urls.py              # 根 URL 配置
  │   ├── wsgi.py              # WSGI 入口
  │   └── asgi.py              # ASGI 入口（异步）
  ├── users/                   # App
  │   ├── models.py            # 数据模型
  │   ├── views.py             # 视图逻辑
  │   ├── urls.py              # URL 路由
  │   ├── serializers.py       # 序列化器（DRF）
  │   ├── admin.py             # Admin 配置
  │   ├── forms.py             # 表单
  │   ├── tests.py             # 测试
  │   ├── signals.py           # 信号
  │   ├── apps.py              # App 配置
  │   └── migrations/          # 数据库迁移
  ├── templates/               # 模板文件
  └── static/                  # 静态文件

请求处理流程：
  1. URL 路由（urls.py）→ 匹配 View
  2. View 处理业务逻辑
  3. Model 与数据库交互
  4. Template 渲染 HTML
  5. 返回 HttpResponse

创建项目和 App：
  # 创建项目
  django-admin startproject myproject
  cd myproject

  # 创建 App
  python manage.py startapp users

  # 注册 App（settings.py）
  INSTALLED_APPS = [
      'django.contrib.admin',
      'django.contrib.auth',
      'users',
  ]

URL 配置：
  # myproject/urls.py
  from django.urls import path, include

  urlpatterns = [
      path('admin/', admin.site.urls),
      path('api/users/', include('users.urls')),
      path('api/posts/', include('posts.urls')),
  ]

  # users/urls.py
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.UserListView.as_view(), name='user-list'),
      path('<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
  ]

  # 带命名空间
  app_name = 'users'
  urlpatterns = [
      path('', views.user_list, name='list'),
  ]
  # 反向解析：reverse('users:list')
```


## Django ORM


```
模型定义：
  from django.db import models
  from django.contrib.auth.models import User

  class Post(models.Model):
      STATUS_CHOICES = [
          ('draft', '草稿'),
          ('published', '已发布'),
      ]

      title = models.CharField(max_length=200)
      content = models.TextField()
      author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
      status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
      tags = models.ManyToManyField('Tag', blank=True)
      created_at = models.DateTimeField(auto_now_add=True)
      updated_at = models.DateTimeField(auto_now=True)
      view_count = models.PositiveIntegerField(default=0)

      class Meta:
          ordering = ['-created_at']
          indexes = [
              models.Index(fields=['-created_at']),
              models.Index(fields=['author', 'status']),
          ]
          verbose_name = '文章'
          verbose_name_plural = '文章'

      def __str__(self):
          return self.title

常用查询操作：
  # 创建
  post = Post.objects.create(title='Hello', content='World', author=user)

  # 查询全部
  posts = Post.objects.all()

  # 过滤
  posts = Post.objects.filter(status='published')
  posts = Post.objects.filter(title__contains='Django')
  posts = Post.objects.filter(created_at__year=2025)
  posts = Post.objects.filter(view_count__gte=100)

  # 排序
  posts = Post.objects.order_by('-created_at')

  # 链式查询
  posts = (Post.objects
      .filter(status='published')
      .select_related('author')         # 预加载外键
      .prefetch_related('tags')         # 预加载多对多
      .order_by('-view_count')[:10])

  # 聚合
  from django.db.models import Count, Avg, Sum, Max
  stats = Post.objects.aggregate(
      total=Count('id'),
      avg_views=Avg('view_count'),
      max_views=Max('view_count'),
  )

  # 分组
  from django.db.models import Count
  author_stats = (Post.objects
      .values('author__username')
      .annotate(post_count=Count('id'))
      .order_by('-post_count'))

  # Q 对象（复杂条件）
  from django.db.models import Q
  posts = Post.objects.filter(
      Q(title__contains='Django') | Q(content__contains='Python'),
      status='published',
  )

  # F 对象（字段间比较）
  from django.db.models import F
  posts = Post.objects.filter(view_count__gt=F('author__post_count'))

  # 原生 SQL
  posts = Post.objects.raw('SELECT * FROM posts WHERE status = %s', ['published'])
  # 或
  from django.db import connection
  with connection.cursor() as cursor:
      cursor.execute("SELECT COUNT(*) FROM posts")
      row = cursor.fetchone()

N+1 查询优化：
  # 错误（N+1）
  for post in Post.objects.all():
      print(post.author.username)  # 每次循环都查询 author

  # 正确（JOIN 查询）
  for post in Post.objects.select_related('author'):
      print(post.author.username)  # 一次性加载

  # 多对多用 prefetch_related
  for post in Post.objects.prefetch_related('tags'):
      print(post.tags.all())  # 一次性加载所有 tags
```


## 数据库迁移（Migrations）


```
迁移命令：
  # 生成迁移文件
  python manage.py makemigrations
  python manage.py makemigrations users    # 指定 App

  # 执行迁移
  python manage.py migrate
  python manage.py migrate users           # 指定 App

  # 查看迁移状态
  python manage.py showmigrations

  # 查看迁移 SQL
  python manage.py sqlmigrate users 0001

  # 回退迁移
  python manage.py migrate users 0002      # 回退到 0002

迁移文件示例：
  # users/migrations/0001_initial.py
  from django.db import migrations, models
  import django.db.models.deletion

  class Migration(migrations.Migration):
      initial = True
      dependencies = [
          ('auth', '0012_user_first_name_max_length'),
      ]
      operations = [
          migrations.CreateModel(
              name='Post',
              fields=[
                  ('id', models.BigAutoField(auto_created=True, primary_key=True)),
                  ('title', models.CharField(max_length=200)),
                  ('content', models.TextField()),
                  ('status', models.CharField(default='draft', max_length=10)),
                  ('created_at', models.DateTimeField(auto_now_add=True)),
                  ('author', models.ForeignKey(
                      on_delete=django.db.models.deletion.CASCADE,
                      to='auth.user',
                  )),
              ],
              options={
                  'ordering': ['-created_at'],
              },
          ),
      ]

数据迁移：
  # 创建空迁移文件
  python manage.py makemigrations users --empty

  # 编写数据迁移
  def migrate_data(apps, schema_editor):
      Post = apps.get_model('users', 'Post')
      Post.objects.filter(status='').update(status='draft')

  class Migration(migrations.Migration):
      dependencies = [('users', '0002')]
      operations = [
          migrations.RunPython(migrate_data, migrations.RunPython.noop),
      ]

迁移注意事项：
  1. 模型修改后必须 makemigrations
  2. 生产环境先备份再 migrate
  3. 不要修改已提交的迁移文件
  4. 冲突解决：python manage.py makemigrations --merge
  5. 大表加字段可能锁表，用 --fake 先跳过
```


## Admin 后台与中间件


```
Admin 配置：
  from django.contrib import admin
  from .models import Post, Tag

  @admin.register(Post)
  class PostAdmin(admin.ModelAdmin):
      list_display = ['title', 'author', 'status', 'created_at', 'view_count']
      list_filter = ['status', 'created_at', 'author']
      search_fields = ['title', 'content']
      list_editable = ['status']
      list_per_page = 20
      date_hierarchy = 'created_at'
      raw_id_fields = ['author']          # 外键搜索选择
      filter_horizontal = ['tags']        # 多对多选择器

      # 自定义操作
      actions = ['make_published']
      def make_published(self, request, queryset):
          queryset.update(status='published')
      make_published.short_description = '标记为已发布'

      # 行内编辑
      class CommentInline(admin.TabularInline):
          model = Comment
          extra = 1

中间件（Middleware）：
  # 自定义中间件
  class RequestLoggingMiddleware:
      def __init__(self, get_response):
          self.get_response = get_response

      def __call__(self, request):
          # 请求前处理
          import time
          start = time.time()

          response = self.get_response(request)

          # 响应后处理
          duration = time.time() - start
          print(f"{request.method} {request.path} - {duration:.3f}s")

          return response

      def process_exception(self, request, exception):
          # 异常处理
          print(f"Exception: {exception}")
          return None

  # 注册（settings.py）
  MIDDLEWARE = [
      'django.middleware.security.SecurityMiddleware',
      'django.contrib.sessions.middleware.SessionMiddleware',
      'django.middleware.common.CommonMiddleware',
      'django.middleware.csrf.CsrfViewMiddleware',
      'django.contrib.auth.middleware.AuthenticationMiddleware',
      'myapp.middleware.RequestLoggingMiddleware',
  ]

信号（Signals）：
  from django.db.models.signals import post_save, pre_delete
  from django.dispatch import receiver
  from django.contrib.auth.models import User

  @receiver(post_save, sender=User)
  def create_user_profile(sender, instance, created, **kwargs):
      if created:
          Profile.objects.create(user=instance)

  @receiver(pre_delete, sender=Post)
  def cleanup_post_files(sender, instance, **kwargs):
      if instance.cover:
          instance.cover.delete(save=False)

类视图（Class-Based Views）：
  from django.views.generic import ListView, DetailView, CreateView
  from django.contrib.auth.mixins import LoginRequiredMixin

  class PostListView(ListView):
      model = Post
      template_name = 'posts/list.html'
      context_object_name = 'posts'
      paginate_by = 20

      def get_queryset(self):
          return Post.objects.filter(status='published').select_related('author')

  class PostCreateView(LoginRequiredMixin, CreateView):
      model = Post
      fields = ['title', 'content', 'tags']
      template_name = 'posts/create.html'
      success_url = '/posts/'

      def form_valid(self, form):
          form.instance.author = self.request.user
          return super().form_valid(form)
```


> **Note:** Django MTV 架构中 Model 负责数据、Template 负责展示、View 负责业务逻辑。ORM 支持链式查询、聚合、Q/F 对象等高级操作，注意用 select_related/prefetch_related 避免 N+1 问题。迁移系统通过 makemigrations 和 migrate 管理数据库变更。Admin 后台、中间件和信号提供了强大的扩展能力。


<!-- Converted from: 01_Django核心概念.html -->
