# Mongoose ODM


## 🔌 Mongoose ODM


Mongoose 安装与连接、Schema 定义与验证、Model CRUD、虚拟属性 virtuals、中间件 hooks (pre/post save)、populate 引用查询。


## Mongoose 基础


```
// ========== Mongoose ==========
// Node.js MongoDB ODM (Object Document Mapping)
// 提供 schema 验证、中间件、populate 等功能

// ========== 安装 ==========
npm install mongoose

// ========== 连接 ==========
const mongoose = require('mongoose');

// 连接数据库
mongoose.connect('mongodb://localhost:27017/mydb')
    .then(() => console.log('Connected'))
    .catch(err => console.error(err));

// 连接选项
mongoose.connect('mongodb://localhost:27017/mydb', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    maxPoolSize: 10,           // 连接池大小
    serverSelectionTimeoutMS: 5000,  // 超时
    heartbeatFrequencyMS: 10000,
});

// 事件监听
mongoose.connection.on('connected', () => {});
mongoose.connection.on('error', (err) => {});
mongoose.connection.on('disconnected', () => {});

// 断开连接
mongoose.disconnect();

// ========== Schema 定义 ==========
const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    age: { type: Number, min: 0, max: 150 },
    status: {
        type: String,
        enum: ['active', 'inactive', 'banned'],
        default: 'active'
    },
    tags: [String],
    address: {
        city: String,
        street: String
    },
    scores: [{
        subject: String,
        score: Number
    }],
    created_at: { type: Date, default: Date.now }
});

// ========== Model ==========
const User = mongoose.model('User', userSchema);
// 'User' → 数据库集合 'users' (自动小写+复数)
```


## CRUD 操作


```
// ========== 创建 ==========

// 方法 1: save()
const user = new User({
    name: "Alice",
    email: "alice@test.com",
    age: 28,
    tags: ["tech", "music"]
});
await user.save();

// 方法 2: create()
const user = await User.create({
    name: "Bob",
    email: "bob@test.com",
    age: 25
});

// 批量创建
const users = await User.insertMany([
    { name: "Carol", email: "carol@test.com" },
    { name: "Dave", email: "dave@test.com" }
]);

// ========== 查询 ==========

// 基本查询
await User.find()                                   // 所有
await User.findById(id)                              // 按 ID
await User.findOne({ email: "alice@test.com" })      // 单条

// 条件查询
await User.find({ age: { $gte: 25 } })
await User.find({ tags: "tech" })
await User.find({ name: /^A/ })

// 链式查询
await User.find({ age: { $gte: 25 } })
    .sort({ age: -1 })
    .skip(10)
    .limit(5)
    .select('name email')         // 只返回指定字段
    .lean()                       // 返回普通 JS 对象 (更快)

// ========== 更新 ==========

// 更新单条
await User.updateOne(
    { email: "alice@test.com" },
    { $set: { age: 29 } }
)

// 更新多条
await User.updateMany(
    { status: "inactive" },
    { $set: { status: "active" } }
)

// 查找并更新 (返回原文档)
const oldUser = await User.findOneAndUpdate(
    { email: "alice@test.com" },
    { $inc: { login_count: 1 } }
)

// 返回更新后的文档
const newUser = await User.findOneAndUpdate(
    { email: "alice@test.com" },
    { $set: { age: 30 } },
    { new: true }       // 返回更新后
)

// ========== 删除 ==========
await User.deleteOne({ email: "bob@test.com" })
await User.deleteMany({ status: "inactive" })
await User.findByIdAndDelete(id)
```


## Schema 进阶


```
// ========== 字段验证 ==========
const productSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, '名称必填'],
        minlength: [2, '名称至少 2 字符'],
        maxlength: [100, '名称最多 100 字符'],
        trim: true,
    },
    price: {
        type: Number,
        required: true,
        min: [0, '价格不能为负'],
    },
    email: {
        type: String,
        match: [/^\S+@\S+\.\S+$/, '邮箱格式无效'],
        lowercase: true,  // 自动转小写
    },
    url: {
        type: String,
        validate: {
            validator: function(v) {
                return v.startsWith('https://');
            },
            message: 'URL 必须以 https:// 开头'
        }
    },
    stock: {
        type: Number,
        default: 0,
        validate: {
            validator: Number.isInteger,
            message: '{VALUE} 不是整数'
        }
    }
});

// ========== 虚拟属性 (virtuals) ==========
// 不存储在数据库, 但可像普通字段一样访问

userSchema.virtual('fullName').get(function() {
    return `${this.firstName} ${this.lastName}`;
});

userSchema.virtual('isAdult').get(function() {
    return this.age >= 18;
});

// 虚拟属性不包含在 JSON 输出中, 需要显式允许:
const userSchema = new mongoose.Schema({...}, {
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// ========== 选项 ==========
const schema = new mongoose.Schema({...}, {
    timestamps: true,        // 自动添加 createdAt/updatedAt
    strict: true,            // 只保存 schema 定义字段
    collection: 'my_users',  // 指定集合名
});
```


## 中间件与 Populate


```
// ========== 中间件 (Hooks) ==========

// pre-save (保存前)
userSchema.pre('save', function(next) {
    // this → 当前文档
    this.updated_at = new Date();

    if (this.isModified('password')) {
        this.password_hash = hashPassword(this.password);
    }
    next();
});

// post-save (保存后)
userSchema.post('save', function(doc) {
    console.log('User saved:', doc.email);
});

// pre-remove
userSchema.pre('remove', async function(next) {
    // 删除关联数据
    await Comment.deleteMany({ author: this._id });
    next();
});

// pre-find (查询前)
userSchema.pre(/^find/, function(next) {
    // this → query object
    this.where({ status: { $ne: 'deleted' } });
    next();
});

// ========== Populate (引用查询) ==========

// 定义引用
const orderSchema = new mongoose.Schema({
    user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    products: [{
        product: { type: mongoose.Schema.Types.ObjectId, ref: 'Product' },
        quantity: Number
    }],
    total: Number
});

const Order = mongoose.model('Order', orderSchema);

// 填充引用字段
const order = await Order.findById(orderId)
    .populate('user', 'name email')        // 填充用户, 只取 name/email
    .populate('products.product', 'name price');  // 填充商品

// 嵌套 populate
const order = await Order.findById(orderId)
    .populate({
        path: 'user',
        populate: { path: 'profile' }
    });

// ========== .lean() 性能优化 ==========
// .lean() 返回普通 JS 对象 (无 Mongoose 文档方法)
// 性能提升: 4-10x 更快

// 不启用 lean:
const users = await User.find({ age: { $gte: 25 } });
// 每个 user 是 Mongoose Document (有 save/remove 等方法)

// 启用 lean:
const users = await User.find({ age: { $gte: 25 } }).lean();
// 每个 user 是普通 JS 对象

// 只读查询推荐使用 .lean()
```


> **Note:** 💡 Mongoose 要点: Schema 定义字段和验证; Model 对应集合; pre/post 中间件在操作前后执行; virtuals 不存数据库; populate 替代 MongoDB $lookup; .lean() 性能优化; timestamps 自动管理时间。


## 练习


<!-- Converted from: 48_Mongoose ODM.html -->
