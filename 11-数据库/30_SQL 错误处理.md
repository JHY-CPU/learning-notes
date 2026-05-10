# SQL 错误处理


## ⚠️ SQL 错误处理


MySQL 错误码、SIGNAL/RESIGNAL 自定义错误、DECLARE HANDLER 异常处理、GET DIAGNOSTICS、存储过程错误处理、事务回滚策略、应用层错误处理。


## MySQL 错误码


```
// ========== 常见错误码 ==========
-- MySQL 使用 SQLSTATE (5位) + 错误号

-- 常见错误:
-- 1045: 拒绝访问 (Access denied)
-- 1046: 没有选择数据库
-- 1048: 列不能为 NULL
-- 1054: 未知列
-- 1062: 唯一键冲突 (Duplicate entry)
-- 1146: 表不存在
-- 1205: 锁等待超时 (Lock wait timeout)
-- 1213: 死锁 (Deadlock)
-- 1451: 外键约束失败
-- 2002: 连接拒绝

-- 查看错误:
SHOW ERRORS;
SHOW WARNINGS;
GET DIAGNOSTICS;  -- MySQL 5.6+

-- ========== 错误处理语法 ==========

-- DECLARE ... HANDLER 语法:
-- DECLARE action HANDLER FOR condition_value statement;

-- action: CONTINUE | EXIT | UNDO
-- condition_value: SQLSTATE | MySQL error code | condition name

-- 1. CONTINUE: 继续执行
-- 2. EXIT: 退出当前程序块
-- 3. UNDO: 不支持

-- ========== 基本错误处理 ==========

DELIMITER $$

CREATE PROCEDURE insert_user(IN p_name VARCHAR(50), IN p_email VARCHAR(100))
BEGIN
    DECLARE EXIT HANDLER FOR 1062  -- 唯一键冲突
    BEGIN
        SELECT CONCAT('Duplicate email: ', p_email) AS error;
        ROLLBACK;
    END;

    DECLARE EXIT HANDLER FOR 1048  -- NOT NULL 违例
    BEGIN
        SELECT 'Name or email cannot be NULL' AS error;
        ROLLBACK;
    END;

    START TRANSACTION;
    INSERT INTO users (name, email) VALUES (p_name, p_email);
    COMMIT;
    SELECT 'User inserted successfully' AS result;
END$$

DELIMITER ;
```


## SIGNAL / RESIGNAL


```
// ========== SIGNAL ==========
-- SIGNAL: 主动抛出错误
-- 可用于存储过程中验证数据

-- 1. 基本 SIGNAL
DELIMITER $$

CREATE PROCEDURE update_price(IN p_id INT, IN p_price DECIMAL(10,2))
BEGIN
    IF p_price < 0 THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'Price cannot be negative';
    END IF;

    IF p_price > 100000 THEN
        SIGNAL SQLSTATE '45001'
            SET MESSAGE_TEXT = 'Price exceeds maximum allowed';
    END IF;

    UPDATE products SET price = p_price WHERE id = p_id;
END$$

DELIMITER ;

-- 2. SIGNAL 带多个属性
SIGNAL SQLSTATE '45000'
    SET MESSAGE_TEXT = 'Validation error',
        MYSQL_ERRNO = 1001,
        TABLE_NAME = 'products';

-- 3. RESIGNAL: 重新抛出错误 (在 handler 中)
DELIMITER $$

CREATE PROCEDURE process_order(IN p_order_id INT)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        -- 记录日志
        INSERT INTO error_log (message, occurred_at)
        VALUES ('Error processing order', NOW());

        -- 重新抛出
        RESIGNAL;
    END;

    -- 业务逻辑...
    UPDATE orders SET status = 'processed' WHERE id = p_order_id;
END$$

DELIMITER ;

-- ========== 自定义错误码 ==========
-- SQLSTATE: 5 位编码
-- 以 '45' 开头的 SQLSTATE 是应用程序错误
-- '45000' - '45999': 自定义错误范围

-- IF condition THEN
--     SIGNAL SQLSTATE '45000'
--         SET MESSAGE_TEXT = 'Custom error message';
-- END IF;
```


## GET DIAGNOSTICS


```
// ========== GET DIAGNOSTICS ==========
-- 获取错误详细信息

-- 1. 获取影响行数
GET DIAGNOSTICS @rows = ROW_COUNT;
SELECT @rows AS affected_rows;

-- 2. 获取错误信息
DELIMITER $$

CREATE PROCEDURE safe_insert(IN p_name VARCHAR(50))
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1
            @errno = MYSQL_ERRNO,
            @sqlstate = RETURNED_SQLSTATE,
            @message = MESSAGE_TEXT;

        SELECT
            @errno AS error_number,
            @sqlstate AS sql_state,
            @message AS error_message;

        ROLLBACK;
    END;

    START TRANSACTION;
    INSERT INTO users (name) VALUES (p_name);
    COMMIT;
END$$

DELIMITER ;

-- ========== 应用层错误处理 (Python) ==========
-- import mysql.connector
-- from mysql.connector import Error
--
-- try:
--     connection = mysql.connector.connect(...)
--     cursor = connection.cursor()
--
--     cursor.execute("INSERT INTO users (name) VALUES (%s)", ("Alice",))
--     connection.commit()
--
-- except mysql.connector.Error as err:
--     print(f"Error: {err}")
--     print(f"Error Code: {err.errno}")
--     print(f"SQLSTATE: {err.sqlstate}")
--     connection.rollback()
--
-- finally:
--     if connection.is_connected():
--         cursor.close()
--         connection.close()

-- ========== 事务回滚策略 ==========
-- 1. 出现错误 → 回滚整个事务
-- 2. 使用 SAVEPOINT → 部分回滚
-- 3. 应用层死锁重试 (最多 3 次)
```


> **Note:** 💡 错误处理: DECLARE HANDLER 捕获错误; SIGNAL 主动抛错; RESIGNAL 重新抛出; GET DIAGNOSTICS 获取详情; 错误码: 1062 唯一冲突, 1213 死锁, 1451 外键; 应用层捕获并处理错误。


## 练习


<!-- Converted from: 30_SQL 错误处理.html -->
