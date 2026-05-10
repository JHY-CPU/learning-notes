# SQL 用户与权限管理


## 🔐 SQL 用户与权限管理


创建/删除用户、GRANT/REVOKE 权限、权限层级、角色管理、MySQL 与 PostgreSQL 权限模型、最小权限原则、最佳实践。


## 用户管理


```
// ========== MySQL 用户管理 ==========

-- 1. 创建用户
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'strong_password_123';
CREATE USER 'readonly_user'@'%' IDENTIFIED BY 'readonly_pass';
CREATE USER 'admin'@'192.168.1.%' IDENTIFIED BY 'admin_pass';

-- 说明:
-- 'user'@'host' 中的 host 限制登录来源
-- 'localhost':   仅本机
-- '%':           任意主机
-- '192.168.1.%': 指定网段

-- 2. 删除用户
DROP USER 'old_user'@'localhost';

-- 3. 修改密码
ALTER USER 'app_user'@'localhost' IDENTIFIED BY 'new_password';

-- 4. 查看用户
SELECT User, Host, authentication_string FROM mysql.user;

-- ========== PostgreSQL 用户管理 ==========
-- PostgreSQL 中用户 = 角色 (ROLE)

-- 创建用户
CREATE ROLE app_user WITH LOGIN PASSWORD 'strong_password';
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'readonly_pass';

-- 创建角色 (无登录权限)
CREATE ROLE developers;

-- 删除用户
DROP ROLE old_user;

-- 修改密码
ALTER ROLE app_user WITH PASSWORD 'new_password';

-- 查看用户
-- \du  (psql 命令)
```


## 权限层级


```
// ========== MySQL 权限层级 ==========
-- 权限从高到低:

-- 1. 全局层级 (*.*)
--    影响所有数据库
--    GRANT SELECT ON *.* TO 'user'@'host';

-- 2. 数据库层级 (db_name.*)
--    影响指定数据库中所有表
--    GRANT SELECT ON mydb.* TO 'user'@'host';

-- 3. 表层级 (db_name.table_name)
--    影响指定表
--    GRANT SELECT ON mydb.users TO 'user'@'host';

-- 4. 列层级 (指定列)
--    影响指定列
--    GRANT SELECT (col1, col2) ON mydb.users TO 'user'@'host';

-- 5. 存储过程层级
--    GRANT EXECUTE ON PROCEDURE mydb.proc_name TO 'user'@'host';

-- ========== 常见权限 ==========
-- SELECT:     查询数据
-- INSERT:     插入数据
-- UPDATE:     更新数据
-- DELETE:     删除数据
-- CREATE:     创建数据库/表
-- DROP:       删除表
-- ALTER:      修改表结构
-- INDEX:      管理索引
-- EXECUTE:    执行存储过程
-- ALL PRIVILEGES: 所有权限
-- GRANT OPTION:   允许转授权限

-- ========== 最小权限示例 ==========
-- 应用用户: 只需要 CRUD
GRANT SELECT, INSERT, UPDATE, DELETE ON mydb.* TO 'app_user'@'localhost';

-- 只读用户: 只允许查询
GRANT SELECT ON mydb.* TO 'readonly_user'@'%';

-- 管理员: 所有权限
GRANT ALL PRIVILEGES ON mydb.* TO 'admin'@'192.168.1.%';

-- 备份用户: LOCK TABLES + SELECT
GRANT SELECT, LOCK TABLES ON *.* TO 'backup_user'@'localhost';

-- 注意: 修改权限后要刷新!
FLUSH PRIVILEGES;
```


## GRANT / REVOKE


```
// ========== MySQL GRANT ==========

-- 1. 授予权限
-- 授予 app_user 对 mydb 所有表的 SELECT/INSERT
GRANT SELECT, INSERT ON mydb.* TO 'app_user'@'localhost';

-- 授予所有数据库的 SELECT 权限 (只读)
GRANT SELECT ON *.* TO 'readonly_user'@'%';

-- 授予列级权限 (只允许查看 email, 不查看 password_hash)
GRANT SELECT (id, username, email) ON mydb.users TO 'support_user'@'localhost';

-- 授予权限并允许转授
GRANT SELECT ON mydb.* TO 'manager'@'localhost' WITH GRANT OPTION;

-- ========== MySQL REVOKE ==========

-- 撤销权限
REVOKE INSERT ON mydb.* FROM 'app_user'@'localhost';

-- 撤销所有权限
REVOKE ALL PRIVILEGES ON mydb.* FROM 'old_user'@'localhost';

-- 查看用户权限
SHOW GRANTS FOR 'app_user'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;

-- ========== PostgreSQL GRANT/REVOKE ==========

-- 授予权限
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- 授予数据库连接权限
GRANT CONNECT ON DATABASE mydb TO app_user;

-- 撤销权限
REVOKE DELETE ON ALL TABLES IN SCHEMA public FROM app_user;

-- 默认权限 (新表自动授权)
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO readonly_user;
```


## 角色管理


```
// ========== MySQL 角色 (8.0+) ==========
-- 角色: 权限的集合,简化权限管理

-- 1. 创建角色
CREATE ROLE 'read_only', 'read_write', 'db_admin';

-- 2. 授予角色权限
GRANT SELECT ON mydb.* TO 'read_only';
GRANT SELECT, INSERT, UPDATE, DELETE ON mydb.* TO 'read_write';
GRANT ALL PRIVILEGES ON mydb.* TO 'db_admin';

-- 3. 创建用户并赋予角色
CREATE USER 'alice'@'localhost' IDENTIFIED BY 'pass123';
CREATE USER 'bob'@'localhost' IDENTIFIED BY 'pass456';
CREATE USER 'admin'@'localhost' IDENTIFIED BY 'admin123';

GRANT 'read_write' TO 'alice'@'localhost';
GRANT 'read_only' TO 'bob'@'localhost';
GRANT 'db_admin' TO 'admin'@'localhost';

-- 4. 设置默认角色 (登录时自动激活)
SET DEFAULT ROLE 'read_write' TO 'alice'@'localhost';

-- 5. 激活角色
SET ROLE 'read_write';
-- 查看当前角色: SELECT CURRENT_ROLE();

-- ========== PostgreSQL 角色 ==========
-- PostgreSQL 角色天然支持继承

-- 创建角色组
CREATE ROLE developers;
CREATE ROLE readers;
CREATE ROLE admins;

-- 授予组权限
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO developers;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readers;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admins;

-- 创建用户并加入角色组
CREATE ROLE alice WITH LOGIN PASSWORD 'pass123' IN ROLE developers;
CREATE ROLE bob WITH LOGIN PASSWORD 'pass456' IN ROLE readers;

-- 或: GRANT developers TO alice;

-- ========== 角色管理优势 ==========
-- 1. 批量管理: 修改角色 → 影响所有成员
-- 2. 层级继承: 角色可以包含角色
-- 3. 职责分离: 开发/运维/只读分离
```


## 安全最佳实践


```
// ========== 数据库安全最佳实践 ==========

-- 1. 最小权限原则
--   只授予完成任务所需的最少权限
--   应用用户: 只给 CRUD, 不给 DDL
--   只读用户: 只给 SELECT

-- 2. 密码安全
--   强密码: 至少 12 位, 字母+数字+符号
--   定期更换
--   不要使用默认密码

-- 3. 网络隔离
--   限制登录来源 (host)
--   应用用户只允许从应用服务器连接
--   禁用 root 远程登录
--   DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1');

-- 4. 权限审计
--   定期检查用户权限
SHOW GRANTS FOR 'app_user'@'localhost';
--   移除不必要的 GRANT OPTION
--   删除无用用户

-- 5. 敏感数据保护
--   不要明文存储密码 (哈希存储)
--   敏感列限制访问 (列级权限)
--   考虑数据加密 (TDE / 列加密)

-- 6. MySQL 安全加固
--   移除匿名用户:
DELETE FROM mysql.user WHERE User = '';
--   设置 root 密码:
ALTER USER 'root'@'localhost' IDENTIFIED BY 'strong_password';
--   禁用 LOAD DATA LOCAL:
--   SET GLOBAL local_infile = 0;

-- 7. 连接安全
--   使用 SSL/TLS 连接
--   设置最大连接数
--   设置超时时间

-- ========== 管理检查清单 ==========
-- ✅ 没有默认密码
-- ✅ 没有匿名用户
-- ✅ root 仅本地登录
-- ✅ 应用用户最小权限
-- ✅ 定期密码轮换
-- ✅ 开启查询日志 (审计用)
-- ✅ 备份权限配置
```


> **Note:** 💡 权限管理要点: 最小权限原则; 区分用户类型 (应用/只读/管理); 角色简化权限管理; 限制登录来源; 定期审计权限; root 仅限本地登录; 权限修改后要 FLUSH PRIVILEGES。


## 练习


<!-- Converted from: 15_SQL 用户与权限管理.html -->
