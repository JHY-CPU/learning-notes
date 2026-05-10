# SQL 视图与存储过程


## 👁️ SQL 视图与存储过程


视图 CREATE VIEW、可更新视图、物化视图、存储过程 CREATE PROCEDURE、参数与变量、游标、函数 CREATE FUNCTION、触发器 CREATE TRIGGER。


## 视图 (View)


```
// ========== 视图概念 ==========
-- 视图: 虚拟表,基于 SELECT 查询
-- 不存储数据,只存储查询定义
-- 每次查询视图时执行底层 SQL

-- 优点:
-- 1. 简化复杂查询 (封装 JOIN/聚合)
-- 2. 安全性 (隐藏敏感列)
-- 3. 逻辑层抽象 (底层表结构变化不影响)

-- ========== 创建视图 ==========
-- 创建活跃用户视图
CREATE VIEW active_users AS
SELECT id, username, email, created_at
FROM users
WHERE is_active = TRUE
  AND deleted_at IS NULL;

-- 使用视图 (就像查询表)
SELECT * FROM active_users ORDER BY created_at DESC;
SELECT username FROM active_users WHERE email LIKE '%@company.com';

-- 创建视图 (订单汇总)
CREATE VIEW order_summary AS
SELECT
    u.id AS user_id,
    u.name AS user_name,
    COUNT(o.id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_spent,
    MAX(o.created_at) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name;

-- 使用汇总视图
SELECT * FROM order_summary WHERE total_spent > 1000;

-- ========== 查看/删除视图 ==========
-- MySQL:
SHOW CREATE VIEW active_users;

-- 删除视图:
DROP VIEW IF EXISTS active_users;
```


## 可更新视图与物化视图


```
// ========== 可更新视图 ==========
-- 某些视图支持 INSERT/UPDATE/DELETE
-- 条件: 不包含 DISTINCT/聚合/GROUP BY/UNION/子查询

-- 简单视图 (可更新)
CREATE VIEW user_contacts AS
SELECT id, username, email, phone
FROM users WHERE is_active = TRUE;

-- 通过视图更新基表
UPDATE user_contacts SET email = 'new@test.com' WHERE id = 1;
-- 等价于: UPDATE users SET email = ... WHERE id = 1 AND is_active = TRUE;

-- 通过视图插入
INSERT INTO user_contacts (username, email) VALUES ('new', 'new@test.com');

-- 带 CHECK OPTION (防止插入后消失)
CREATE VIEW vip_users AS
SELECT * FROM users WHERE total_spent > 1000
WITH CHECK OPTION;
-- INSERT 必须满足 total_spent > 1000,否则报错

-- ========== 物化视图 (Materialized View) ==========
-- PostgreSQL / Oracle / MySQL 不支持!
-- 实际存储数据 (不是虚拟表)
-- 需要手动刷新

-- PostgreSQL:
CREATE MATERIALIZED VIEW mv_order_stats AS
SELECT user_id, COUNT(*) AS orders, SUM(amount) AS total
FROM orders GROUP BY user_id;

-- 查询物化视图
SELECT * FROM mv_order_stats;

-- 刷新物化视图
REFRESH MATERIALIZED VIEW mv_order_stats;

-- ========== 视图 vs 物化视图 ==========
-- 视图:       不存数据,每次执行查询,总是最新
-- 物化视图:   存储数据,需要刷新,不阻塞查询
```


## 存储过程


```
// ========== 存储过程 ==========
-- 存储在数据库中的程序 (预编译 SQL)
-- 支持: 参数 / 变量 / 条件 / 循环 / 游标

-- ========== MySQL 存储过程 ==========
DELIMITER $$

-- 1. 基本存储过程: 获取用户订单数
CREATE PROCEDURE GetUserOrderCount(IN userId INT, OUT orderCount INT)
BEGIN
    SELECT COUNT(*) INTO orderCount
    FROM orders
    WHERE user_id = userId;
END$$

-- 调用:
CALL GetUserOrderCount(1, @cnt);
SELECT @cnt;  -- 输出结果

-- 2. 带条件逻辑
CREATE PROCEDURE ApplyDiscount(IN productId INT, IN discountPct DECIMAL(5,2))
BEGIN
    DECLARE currentPrice DECIMAL(10,2);

    SELECT price INTO currentPrice FROM products WHERE id = productId;

    IF currentPrice IS NOT NULL THEN
        UPDATE products
        SET price = price * (1 - discountPct / 100)
        WHERE id = productId;
        SELECT CONCAT('Price updated from ', currentPrice, ' to ', price * (1 - discountPct / 100)) AS result;
    ELSE
        SELECT 'Product not found' AS result;
    END IF;
END$$

-- 3. 带事务的存储过程
CREATE PROCEDURE TransferMoney(
    IN fromAccount INT,
    IN toAccount INT,
    IN amount DECIMAL(10,2)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'Transaction failed' AS result;
    END;

    START TRANSACTION;

    UPDATE accounts SET balance = balance - amount WHERE id = fromAccount;
    UPDATE accounts SET balance = balance + amount WHERE id = toAccount;

    COMMIT;
    SELECT 'Transfer successful' AS result;
END$$

DELIMITER ;

-- 调用:
CALL TransferMoney(1, 2, 100.00);

-- ========== PostgreSQL 存储过程 (PROCEDURE) ==========
-- CREATE PROCEDURE transfer(from_acc INT, to_acc INT, amount DECIMAL)
-- LANGUAGE plpgsql
-- AS $$
-- BEGIN
--     UPDATE accounts SET balance = balance - amount WHERE id = from_acc;
--     UPDATE accounts SET balance = balance + amount WHERE id = to_acc;
-- END;
-- $$;
```


## 存储函数 (Stored Function)


```
// ========== MySQL 函数 ==========
-- 与存储过程的区别: 函数有返回值,可在 SQL 中使用

DELIMITER $$

CREATE FUNCTION GetUserFullName(userId INT)
RETURNS VARCHAR(100)
DETERMINISTIC   -- 相同输入 → 相同输出
READS SQL DATA  -- 函数只读数据
BEGIN
    DECLARE fullName VARCHAR(100);

    SELECT CONCAT(first_name, ' ', last_name) INTO fullName
    FROM users WHERE id = userId;

    RETURN fullName;
END$$

DELIMITER ;

-- 在 SQL 中使用函数
SELECT id, GetUserFullName(id) AS full_name FROM users;

-- ========== 常用函数示例 ==========

-- 计算年龄
CREATE FUNCTION CalcAge(birthDate DATE)
RETURNS INT
DETERMINISTIC
BEGIN
    RETURN TIMESTAMPDIFF(YEAR, birthDate, CURDATE());
END;

-- 格式化金额
CREATE FUNCTION FormatCurrency(amount DECIMAL(12,2))
RETURNS VARCHAR(20)
DETERMINISTIC
BEGIN
    RETURN CONCAT('¥', FORMAT(amount, 2));
END;

-- ========== 函数 vs 存储过程 ==========
-- 函数: 有返回值,可在 SQL 中直接使用
-- 存储过程: 可有 IN/OUT/INOUT,用 CALL 调用

-- 函数限制:
-- - 不能使用事务 (START/COMMIT/ROLLBACK)
-- - 不能返回结果集 (MySQL)
```


## 触发器 (Trigger)


```
// ========== 触发器概念 ==========
-- 触发器: 在表上发生 INSERT/UPDATE/DELETE 时自动执行

-- 使用场景:
-- 1. 自动记录审计日志 (谁改了什么)
-- 2. 数据验证 (无法用约束表达的复杂规则)
-- 3. 冗余字段更新 (如 order_count)
-- 4. 级联操作 (特殊业务逻辑)

-- ========== MySQL 触发器 ==========

-- 1. 审计日志触发器
CREATE TABLE audit_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50),
    action VARCHAR(10),
    record_id INT,
    old_data JSON,
    new_data JSON,
    changed_by VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER trg_users_audit
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (table_name, action, record_id, old_data, new_data, changed_by)
    VALUES (
        'users',
        'UPDATE',
        NEW.id,
        JSON_OBJECT('username', OLD.username, 'email', OLD.email, 'is_active', OLD.is_active),
        JSON_OBJECT('username', NEW.username, 'email', NEW.email, 'is_active', NEW.is_active),
        CURRENT_USER()
    );
END;

-- 2. 自动更新冗余字段
CREATE TRIGGER trg_orders_after_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE users
    SET total_orders = total_orders + 1,
        total_spent = total_spent + NEW.amount
    WHERE id = NEW.user_id;
END;

-- 3. 防止删除 (保护性触发器)
CREATE TRIGGER trg_prevent_admin_delete
BEFORE DELETE ON users
FOR EACH ROW
BEGIN
    IF OLD.role = 'admin' THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Cannot delete admin users';
    END IF;
END;

-- ========== 查看/删除触发器 ==========
SHOW TRIGGERS;
DROP TRIGGER IF EXISTS trg_users_audit;

-- ========== 注意事项 ==========
-- 1. 触发器不可见 → 调试困难
-- 2. 过多触发器影响性能
-- 3. 递归触发器可能死循环
-- 4. 触发器中的错误会回滚主操作
```


> **Note:** 💡 视图/存储过程要点: 视图是虚拟表,简化查询; 物化视图存数据,定期刷新; 存储过程预编译,适合复杂业务逻辑; 函数有返回值,可在 SQL 中调用; 触发器自动响应数据变更,但谨慎使用避免隐式问题。


## 练习


<!-- Converted from: 9_SQL 视图与存储过程.html -->
