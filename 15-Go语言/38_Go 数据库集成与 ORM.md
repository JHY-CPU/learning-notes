# Go 数据库集成与 ORM


## 🗄️ Go 数据库集成与 ORM


database/sql 原生操作、连接池配置、GORM 模型与 CRUD、迁移与钩子、事务、原生 SQL 与性能优化。


## database/sql 基础


```
// ========== 连接数据库 (PostgreSQL) ==========
// import "database/sql"
// import _ "github.com/lib/pq"  // 驱动注册

type DB struct {
    *sql.DB
}

func NewDB(dsn string) (*DB, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("open db: %w", err)
    }

    // 验证连接
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("ping db: %w", err)
    }

    return &DB{db}, nil
}

// ========== 连接池配置 ==========
func configurePool(db *sql.DB) {
    db.SetMaxOpenConns(25)             // 最大打开连接数
    db.SetMaxIdleConns(10)             // 最大空闲连接数
    db.SetConnMaxLifetime(5 * time.Minute)  // 连接最大存活时间
    db.SetConnMaxIdleTime(2 * time.Minute)  // 空闲最大存活时间
}

// 选择准则:
// MaxOpenConns: CPU 核数 * 2 + 有效磁盘数
// MaxIdleConns: 通常 ≤ MaxOpenConns
// ConnMaxLifetime: 避免长时间连接被 LB 断开

// ========== CRUD 操作 ==========
type User struct {
    ID        int
    Name      string
    Email     string
    CreatedAt time.Time
}

// Create
func (db *DB) CreateUser(u *User) error {
    query := `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id, created_at`
    return db.QueryRow(query, u.Name, u.Email).Scan(&u.ID, &u.CreatedAt)
}

// Read (单行)
func (db *DB) GetUser(id int) (*User, error) {
    query := `SELECT id, name, email, created_at FROM users WHERE id = $1`
    u := &User{}
    err := db.QueryRow(query, id).Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt)
    if err == sql.ErrNoRows {
        return nil, fmt.Errorf("user %d not found", id)
    }
    return u, err
}

// Read (多行)
func (db *DB) ListUsers(limit, offset int) ([]User, error) {
    query := `SELECT id, name, email, created_at FROM users ORDER BY id LIMIT $1 OFFSET $2`
    rows, err := db.Query(query, limit, offset)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()  // 检查迭代错误
}

// Update
func (db *DB) UpdateUser(u *User) error {
    query := `UPDATE users SET name=$1, email=$2 WHERE id=$3`
    result, err := db.Exec(query, u.Name, u.Email, u.ID)
    if err != nil {
        return err
    }
    n, _ := result.RowsAffected()
    if n == 0 {
        return fmt.Errorf("user %d not found", u.ID)
    }
    return nil
}

// Delete
func (db *DB) DeleteUser(id int) error {
    query := `DELETE FROM users WHERE id = $1`
    result, err := db.Exec(query, id)
    if err != nil {
        return err
    }
    n, _ := result.RowsAffected()
    if n == 0 {
        return fmt.Errorf("user %d not found", id)
    }
    return nil
}
```


## 事务与批量操作


```
// ========== 事务 ==========
func (db *DB) CreateUserWithProfile(u *User, p *Profile) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()  // 提交后 Rollback 无效

    // 插入用户
    err = tx.QueryRow(
        `INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id`,
        u.Name, u.Email,
    ).Scan(&u.ID)
    if err != nil {
        return fmt.Errorf("insert user: %w", err)
    }

    // 插入 profile
    _, err = tx.Exec(
        `INSERT INTO profiles (user_id, bio, avatar) VALUES ($1, $2, $3)`,
        u.ID, p.Bio, p.Avatar,
    )
    if err != nil {
        return fmt.Errorf("insert profile: %w", err)
    }

    return tx.Commit()
}

// ========== 批量插入 ==========
func (db *DB) BatchCreateUsers(users []User) error {
    tx, err := db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()

    stmt, err := tx.Prepare(`INSERT INTO users (name, email) VALUES ($1, $2)`)
    if err != nil {
        return err
    }
    defer stmt.Close()

    for _, u := range users {
        if _, err := stmt.Exec(u.Name, u.Email); err != nil {
            return err
        }
    }

    return tx.Commit()
}

// ========== 预处理语句 ==========
// PreparedStatement 提高重复查询性能

type UserRepo struct {
    db      *sql.DB
    getStmt *sql.Stmt
}

func NewUserRepo(db *sql.DB) (*UserRepo, error) {
    getStmt, err := db.Prepare(`SELECT id, name, email FROM users WHERE id = $1`)
    if err != nil {
        return nil, err
    }
    return &UserRepo{db: db, getStmt: getStmt}, nil
}

func (r *UserRepo) Get(id int) (*User, error) {
    u := &User{}
    err := r.getStmt.QueryRow(id).Scan(&u.ID, &u.Name, &u.Email)
    return u, err
}

// ========== NULL 值处理 ==========
// sql.NullString, sql.NullInt64, sql.NullFloat64, sql.NullTime

type Product struct {
    ID          int
    Name        string
    Description sql.NullString  // 可能为 NULL
    Price       float64
    CategoryID  sql.NullInt64   // 可能为 NULL
}

func (db *DB) GetProduct(id int) (*Product, error) {
    query := `SELECT id, name, description, price, category_id FROM products WHERE id = $1`
    p := ∏{}
    err := db.QueryRow(query, id).Scan(
        &p.ID, &p.Name, &p.Description, &p.Price, &p.CategoryID,
    )
    if p.Description.Valid {
        fmt.Println("描述:", p.Description.String)
    }
    return p, err
}

// ========== 命名参数 ==========
// PostgreSQL 用 $1, $2, ...
// MySQL 用 ?, ?, ...
// SQLite 用 ? 或 $param

func (db *DB) GetUsersByFilter(name string, minAge int) ([]User, error) {
    query := `SELECT id, name, email FROM users WHERE name ILIKE $1 AND age >= $2`
    rows, err := db.Query(query, "%"+name+"%", minAge)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        rows.Scan(&u.ID, &u.Name, &u.Email)
        users = append(users, u)
    }
    return users, nil
}
```


## GORM 入门


```
// ========== GORM 模型 ==========
// import "gorm.io/gorm"
// import "gorm.io/driver/postgres"

type Product struct {
    gorm.Model                   // 内嵌: ID, CreatedAt, UpdatedAt, DeletedAt
    Code       string            `gorm:"column:code;uniqueIndex;not null"`
    Name       string            `gorm:"type:varchar(200);not null"`
    Price      float64           `gorm:"type:decimal(10,2);default:0"`
    CategoryID uint              `gorm:"index"`
    Category   Category          `gorm:"foreignKey:CategoryID"`
    Status     string            `gorm:"default:active"`
    Tags       []Tag             `gorm:"many2many:product_tags;"`
}

type Category struct {
    ID       uint      `gorm:"primaryKey"`
    Name     string    `gorm:"size:100;not null"`
    Products []Product `gorm:"foreignKey:CategoryID"`
}

// ========== 连接与迁移 ==========
func initDB() (*gorm.DB, error) {
    dsn := "host=localhost user=postgres password=pass dbname=test port=5432 sslmode=disable"
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info),  // SQL 日志
        SkipDefaultTransaction: true,  // 提高性能
        PrepareStmt: true,             // 缓存预处理语句
    })
    if err != nil {
        return nil, err
    }

    // 自动迁移
    db.AutoMigrate(∏{}, &Category{}, &Tag{})

    // 连接池 (通过 sql.DB)
    sqlDB, _ := db.DB()
    sqlDB.SetMaxOpenConns(25)
    sqlDB.SetMaxIdleConns(10)
    sqlDB.SetConnMaxLifetime(5 * time.Minute)

    return db, nil
}

// ========== GORM CRUD ==========
// Create
func createProduct(db *gorm.DB) {
    p := Product{Code: "P001", Name: "笔记本", Price: 99.9, CategoryID: 1}
    result := db.Create(&p)
    fmt.Println(p.ID, result.Error, result.RowsAffected)

    // 批量
    products := []Product{
        {Code: "P002", Name: "鼠标", Price: 29.9},
        {Code: "P003", Name: "键盘", Price: 199},
    }
    db.Create(&products)
}

// Query
func queryProduct(db *gorm.DB) {
    // 单条
    var p Product
    db.First(&p, 1)                          // WHERE id = 1
    db.First(&p, "code = ?", "P001")         // WHERE code = 'P001'
    db.Take(&p)                              // 随机一条
    db.Last(&p)                              // 最后一条

    // 多条
    var products []Product
    db.Where("price > ?", 50).Find(&products)
    db.Where("status = ?", "active").Order("price DESC").Limit(10).Find(&products)

    // 条件
    db.Where(∏{Status: "active", CategoryID: 1}).Find(&products)
    db.Where(map[string]interface{}{"status": "active", "category_id": 1}).Find(&products)
    db.Where([]int{1, 2, 3}).Find(&products)  // WHERE id IN (1,2,3)
}

// Update
func updateProduct(db *gorm.DB) {
    // 更新单个字段
    db.Model(∏{}).Where("id = ?", 1).Update("price", 129.9)

    // 更新多个字段
    db.Model(∏{}).Where("id = ?", 1).Updates(Product{Price: 129.9, Status: "inactive"})

    // 用 map (零值也能更新)
    db.Model(∏{}).Where("id = ?", 1).Updates(map[string]interface{}{
        "price":  129.9,
        "status": "inactive",
    })
}

// Delete
func deleteProduct(db *gorm.DB) {
    // 软删除 (有 gorm.DeletedAt)
    db.Delete(∏{}, 1)

    // 永久删除
    db.Unscoped().Delete(∏{}, 1)
}
```


## GORM 高级


```
// ========== 关联查询 ==========
// Preload 预加载
func preloadExample(db *gorm.DB) {
    var products []Product

    // 预加载关联
    db.Preload("Category").Preload("Tags").Find(&products)

    // 带条件预加载
    db.Preload("Category", "status = ?", "active").Find(&products)

    // 嵌套预加载
    db.Preload("Category.Products").Find(&products)
    for _, p := range products {
        fmt.Println(p.Name, p.Category.Name)
    }
}

// Joins 查询 (减少查询数)
func joinsExample(db *gorm.DB) {
    var results []struct {
        ProductName  string
        CategoryName string
        Price        float64
    }
    db.Model(∏{}).
        Select("products.name as product_name, categories.name as category_name, products.price").
        Joins("left join categories on categories.id = products.category_id").
        Where("products.price > ?", 50).
        Scan(&results)
}

// ========== 钩子 (Hooks) ==========
func (u *User) BeforeCreate(tx *gorm.DB) error {
    u.CreatedAt = time.Now()
    if u.Password != "" {
        hashed, _ := bcrypt.GenerateFromPassword([]byte(u.Password), bcrypt.DefaultCost)
        u.Password = string(hashed)
    }
    return nil
}

func (u *User) AfterFind(tx *gorm.DB) error {
    // 查询后处理
    return nil
}

// 可用钩子:
// BeforeSave / BeforeCreate / BeforeUpdate / BeforeDelete
// AfterSave / AfterCreate / AfterUpdate / AfterDelete / AfterFind

// ========== 事务 ==========
func transactionExample(db *gorm.DB) error {
    return db.Transaction(func(tx *gorm.DB) error {
        if err := tx.Create(∏{Code: "P004", Name: "显示器"}).Error; err != nil {
            return err  // 回滚
        }
        if err := tx.Create(&Category{Name: "外设"}).Error; err != nil {
            return err  // 回滚
        }
        return nil  // 提交
    })
}

// ========== 原生 SQL ==========
func rawSQLExample(db *gorm.DB) {
    type Stats struct {
        CategoryName string
        ProductCount int64
        TotalPrice   float64
    }

    var stats []Stats
    db.Raw(`
        SELECT c.name as category_name,
               COUNT(p.id) as product_count,
               COALESCE(SUM(p.price), 0) as total_price
        FROM categories c
        LEFT JOIN products p ON p.category_id = c.id
        GROUP BY c.id, c.name
    `).Scan(&stats)
}

// ========== 作用域 (Scopes) ==========
func ActiveScope(db *gorm.DB) *gorm.DB {
    return db.Where("status = ?", "active")
}

func PriceAbove(price float64) func(db *gorm.DB) *gorm.DB {
    return func(db *gorm.DB) *gorm.DB {
        return db.Where("price > ?", price)
    }
}

// 使用:
// db.Scopes(ActiveScope, PriceAbove(100)).Find(&products)

// ========== 性能优化 ==========
// 1. SkipDefaultTransaction: true
// 2. PrepareStmt: true
// 3. 只查需要的字段: db.Select("id, name").Find(&products)
// 4. 批量插入分块: db.CreateInBatches(products, 100)
// 5. 不使用循环创建
// 6. 嵌套查询用 Preload + Select
```


> **Note:** 💡 数据库要点: database/sql + 驱动 (lib/pq/pgx); 连接池 SetMaxOpenConns/SetMaxIdleConns/ConnMaxLifetime; QueryRow 单行 / Query 多行 / Exec 写操作; 事务 Begin/Commit/Rollback; sql.NullString 处理 NULL; PreparedStatement 提高性能; GORM 模型标签 gorm:"column/type/index"; AutoMigrate 自动迁移; Preload/Joins 关联查询; 钩子 BeforeCreate/AfterFind; 事务 Transaction; 原生 SQL Raw/Scan; Scopes 复用查询条件。


## 练习


<!-- Converted from: 38_Go 数据库集成与 ORM.html -->
