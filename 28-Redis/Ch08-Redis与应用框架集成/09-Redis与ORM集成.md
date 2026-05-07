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
4. **过期时间**：合理设置缓存TTL
