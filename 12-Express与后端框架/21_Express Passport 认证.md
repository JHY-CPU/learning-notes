# Express Passport 认证


## 🛂 Express Passport 认证


Passport.js 策略模式 (local/jwt/oauth)、serializeUser/deserializeUser、local 策略用户名密码登录、JWT 策略 Bearer Token、Google/GitHub OAuth 登录、多策略组合。


## Passport 概述


```
// ========== Passport.js ==========
// Node.js 认证中间件
// 策略模式: 支持 500+ 认证策略

// ========== 安装 ==========
npm install passport passport-local passport-jwt

// 社交登录:
npm install passport-google-oauth20 passport-github2

// ========== 概念 ==========
// 1. Strategy — 认证策略 (local/jwt/oauth)
// 2. serializeUser — 用户序列化 (存什么到 session)
// 3. deserializeUser — 反序列化 (从 session 还原)
// 4. req.user — 认证后用户对象

// ========== 初始化 ==========
const passport = require('passport');

// 初始化 (Session 模式):
app.use(passport.initialize());
app.use(passport.session());  // 如果使用 session

// 序列化:
passport.serializeUser((user, done) => {
    done(null, user.id);  // 只存 ID 到 session
});

passport.deserializeUser(async (id, done) => {
    try {
        const user = await User.findById(id);
        done(null, user);   // 每次请求还原 user 到 req.user
    } catch (err) {
        done(err);
    }
});
```


## Local 策略


```
// ========== Passport Local Strategy ==========
// 用户名/密码认证

// strategies/local.js:
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;

passport.use(new LocalStrategy(
    {
        usernameField: 'email',      // 默认 username, 改为 email
        passwordField: 'password',
    },
    async (email, password, done) => {
        try {
            const user = await User.findOne({ email }).select('+password');

            if (!user) {
                return done(null, false, { message: 'Invalid credentials' });
            }

            if (!user.isActive) {
                return done(null, false, { message: 'Account deactivated' });
            }

            const isMatch = await bcrypt.compare(password, user.password);
            if (!isMatch) {
                return done(null, false, { message: 'Invalid credentials' });
            }

            return done(null, user);  // 成功 → req.user = user
        } catch (err) {
            return done(err);
        }
    }
));

// ========== 路由使用 ==========
const passport = require('passport');

// 登录路由:
router.post('/login', (req, res, next) => {
    passport.authenticate('local', (err, user, info) => {
        if (err) return next(err);
        if (!user) {
            return res.status(401).json({ message: info.message });
        }

        // 登录成功, 建立 session
        req.logIn(user, (err) => {
            if (err) return next(err);
            return res.json({
                success: true,
                data: user.toPublicJSON(),
            });
        });
    })(req, res, next);
});

// ========== 认证中间件 ==========
function ensureAuthenticated(req, res, next) {
    if (req.isAuthenticated()) {  // Passport 提供的方法
        return next();
    }
    res.status(401).json({ message: 'Authentication required' });
}

router.get('/profile', ensureAuthenticated, (req, res) => {
    res.json(req.user);
});
```


## JWT 策略 (API 模式)


```
// ========== Passport JWT Strategy ==========
// 适合 REST API (无 session, Bearer Token)

// strategies/jwt.js:
const passport = require('passport');
const JwtStrategy = require('passport-jwt').Strategy;
const ExtractJwt = require('passport-jwt').ExtractJwt;

const opts = {
    jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
    secretOrKey: process.env.JWT_SECRET,
    // 从请求头取 token:
    // ExtractJwt.fromAuthHeaderAsBearerToken()
    // 从 Cookie 取:
    // ExtractJwt.fromExtractors([(req) => req.cookies?.token])
};

passport.use(new JwtStrategy(opts, async (payload, done) => {
    try {
        const user = await User.findById(payload.sub);

        if (!user) {
            return done(null, false);
        }

        // 检查密码是否在该 JWT 签发后修改过
        if (user.passwordChangedAt &&
            payload.iat < user.passwordChangedAt.getTime() / 1000) {
            return done(null, false);  // 令牌已失效
        }

        return done(null, user);  // req.user = user
    } catch (err) {
        return done(err, false);
    }
}));

// ========== 路由使用 ==========
// 不需要 session:
app.use(passport.initialize());
// 不用 passport.session()

// 认证中间件:
const authenticate = passport.authenticate('jwt', { session: false });

router.get('/profile', authenticate, (req, res) => {
    res.json(req.user);
});

// ========== 多策略组合 ==========
// 同时支持 local (session) 和 jwt (API):

passport.use('local', new LocalStrategy({ ... }));
passport.use('jwt', new JwtStrategy({ ... }));

// 不同路由用不同策略:
app.post('/login', passport.authenticate('local', { session: true }));
app.get('/api/me', passport.authenticate('jwt', { session: false }), handler);
```


## OAuth 社交登录


```
// ========== Google OAuth ==========
// strategies/google.js:
const GoogleStrategy = require('passport-google-oauth20').Strategy;

passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: '/auth/google/callback',
    scope: ['profile', 'email'],
}, async (accessToken, refreshToken, profile, done) => {
    try {
        // 查找已有用户
        let user = await User.findOne({ googleId: profile.id });

        if (!user) {
            // 或通过 email 查找
            user = await User.findOne({ email: profile.emails[0].value });

            if (user) {
                // 关联 Google ID
                user.googleId = profile.id;
                await user.save();
            } else {
                // 创建新用户
                user = await User.create({
                    name: profile.displayName,
                    email: profile.emails[0].value,
                    googleId: profile.id,
                    avatar: profile.photos[0].value,
                    isVerified: true,
                });
            }
        }

        return done(null, user);
    } catch (err) {
        return done(err, null);
    }
}));

// ========== 路由 ==========
// 发起 Google 登录:
router.get('/auth/google',
    passport.authenticate('google', { scope: ['profile', 'email'] })
);

// Google 回调:
router.get('/auth/google/callback',
    passport.authenticate('google', {
        successRedirect: '/dashboard',
        failureRedirect: '/login',
    })
);

// ========== GitHub OAuth ==========
// 类似 Google, 使用 passport-github2:
passport.use(new GitHubStrategy({
    clientID: process.env.GITHUB_CLIENT_ID,
    clientSecret: process.env.GITHUB_CLIENT_SECRET,
    callbackURL: '/auth/github/callback',
}, async (accessToken, refreshToken, profile, done) => {
    // ... 类似 Google
}));
```


> **Note:** 💡 Passport 要点: 策略模式 (500+ 策略); serializeUser/deserializeUser 管理 session; Local 策略用户名密码; JWT 策略 API 无状态; OAuth 社交登录 Google/GitHub; 多策略可组合; passport.authenticate 中间件; req.login()/req.logout() 管理 session; done(err, user, info) 回调约定。


## 练习


<!-- Converted from: 21_Express Passport 认证.html -->
