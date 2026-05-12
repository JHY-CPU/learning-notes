# Redis与ORM集成

## 一、概念说明

Redis作为缓存层与ORM（如JPA、Hibernate、Django ORM）集成，减少数据库查询。

## 二、Spring Data JPA + Redis

```java
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        // 1. 查缓存
        User user = (User) redisTemplate.opsForValue().get(key);
        if (user != null) return user;
        
        // 2. 查数据库
        user = userRepository.findById(id).orElse(null);
        if (user != null) {
            redisTemplate.opsForValue().set(key, user, 30, TimeUnit.MINUTES);
        }
        return user;
    }
    
    @CacheEvict(value = "user", key = "#id")
    public User updateUser(Long id, UserDTO dto) {
        User user = userRepository.save(dto);
        redisTemplate.delete("user:" + id);
        return user;
    }
}
```

## 三、Django + Redis

```python
from django.core.cache import cache
from .models import User

def get_user(user_id):
    # 1. 查缓存
    user = cache.get(f'user:{user_id}')
    if user:
        return user
    
    # 2. 查数据库
    user = User.objects.get(id=user_id)
    cache.set(f'user:{user_id}', user, 1800)
    return user
```

## 四、缓存模式

```bash
# Cache-Aside（推荐）
# 读：先读缓存，未命中读DB，写入缓存
# 写：更新DB，删除缓存

# Write-Through
# 写：同时更新缓存和DB

# Write-Behind
# 写：更新缓存，异步更新DB
```

## 五、注意事项

1. **缓存一致性**：先更新DB再删缓存
2. **缓存穿透**：空值缓存
3. **序列化**：ORM对象的序列化
## 六、缓存注解详解

```java
// @Cacheable - 查询时缓存
@Cacheable(value = "users", key = "#id", unless = "#result == null")
public User getUser(Long id) {
    return userRepository.findById(id).orElse(null);
}

// @CachePut - 更新时刷新缓存
@CachePut(value = "users", key = "#user.id")
public User updateUser(User user) {
    return userRepository.save(user);
}

// @CacheEvict - 删除时清除缓存
@CacheEvict(value = "users", key = "#id")
public void deleteUser(Long id) {
    userRepository.deleteById(id);
}

// @Caching - 组合注解
@Caching(
    put = { @CachePut(value = "users", key = "#user.id") },
    evict = { @CacheEvict(value = "userList", allEntries = true) }
)
public User saveUser(User user) {
    return userRepository.save(user);
}
```

## 七、Django缓存集成

```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'TIMEOUT': 300,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# views.py
from django.core.cache import cache

def get_user(request, user_id):
    cache_key = f'user:{user_id}'
    
    # 查缓存
    user = cache.get(cache_key)
    if user:
        return JsonResponse(user)
    
    # 查数据库
    user = User.objects.get(id=user_id)
    
    # 写缓存
    cache.set(cache_key, user, timeout=1800)
    
    return JsonResponse(user)

# 使用装饰器
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # 缓存15分钟
def product_list(request):
    products = Product.objects.all()
    return render(request, 'products.html', {'products': products})
```

## 八、缓存一致性方案

```python
class CacheConsistencyManager:
    """缓存一致性管理"""
    
    def __init__(self, redis_client):
        self.r = redis_client
    
    def update_with_consistency(self, key, db_update_func, db_query_func, ttl=3600):
        """保证一致性的更新"""
        # 1. 删除缓存
        self.r.delete(key)
        
        # 2. 更新数据库
        result = db_update_func()
        
        # 3. 延迟删除（处理并发读）
        def delayed_delete():
            time.sleep(0.5)
            self.r.delete(key)
        
        threading.Thread(target=delayed_delete, daemon=True).start()
        
        return result
    
    def read_with_consistency(self, key, db_query_func, ttl=3600):
        """保证一致性的读取"""
        # 1. 查缓存
        cached = self.r.get(key)
        if cached:
            return json.loads(cached)
        
        # 2. 查数据库
        data = db_query_func()
        
        # 3. 写缓存
        if data:
            self.r.setex(key, ttl, json.dumps(data))
        
        return data
```
