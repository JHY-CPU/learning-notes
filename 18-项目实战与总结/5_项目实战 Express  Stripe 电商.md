# 项目实战 Express + Stripe 电商


## 📦 项目实战 6: Express + Stripe 电商


商品与分类、Redis 购物车、Stripe 支付 Intent、Webhook 订单状态更新、PostgreSQL 事务库存一致性。


## 项目结构


```
// ecommerce/
// ├── src/
// │   ├── config/
// │   │   └── index.js
// │   ├── db/
// │   │   ├── knex.js
// │   │   └── migrations/
// │   ├── middleware/
// │   │   ├── auth.js
// │   │   └── validate.js
// │   ├── routes/
// │   │   ├── products.js
// │   │   ├── cart.js
// │   │   ├── checkout.js
// │   │   └── webhook.js
// │   ├── services/
// │   │   ├── stripe.js
// │   │   ├── cart.js
// │   │   └── order.js
// │   └── app.js
// ├── docker-compose.yml
// └── package.json

// ========== 数据库 Schema ==========
// -- migrations/001_create_tables.js
// exports.up = function(knex) {
//   return knex.schema
//     .createTable('products', table => {
//       table.increments('id');
//       table.string('name').notNullable();
//       table.text('description');
//       table.decimal('price', 10, 2).notNullable();
//       table.string('image_url');
//       table.integer('stock').notNullable().defaultTo(0);
//       table.timestamps(true, true);
//     })
//     .createTable('orders', table => {
//       table.increments('id');
//       table.integer('user_id').references('id').inTable('users');
//       table.string('stripe_payment_intent_id');
//       table.string('status').defaultTo('pending');
//       // pending | paid | fulfilled | cancelled
//       table.decimal('total', 10, 2).notNullable();
//       table.timestamps(true, true);
//     })
//     .createTable('order_items', table => {
//       table.increments('id');
//       table.integer('order_id').references('id').inTable('orders');
//       table.integer('product_id').references('id').inTable('products');
//       table.integer('quantity').notNullable();
//       table.decimal('price', 10, 2).notNullable();
//     });
// };
```


## Redis 购物车


```
// ========== Redis 购物车 ==========
// services/cart.js
const redis = require('../db/redis');

const CART_PREFIX = 'cart:';
const CART_TTL = 60 * 60 * 24 * 7; // 7天

class CartService {
  // 获取购物车
  async getCart(userId) {
    const items = await redis.hgetall(`${CART_PREFIX}${userId}`);
    return Object.entries(items || {}).map(([productId, quantity]) => ({
      productId: parseInt(productId),
      quantity: parseInt(quantity),
    }));
  }

  // 添加商品
  async addItem(userId, productId, quantity = 1) {
    const key = `${CART_PREFIX}${userId}`;
    const currentQty = parseInt(await redis.hget(key, productId)) || 0;
    await redis.hset(key, productId, currentQty + quantity);
    await redis.expire(key, CART_TTL);
    return this.getCart(userId);
  }

  // 更新数量
  async updateItem(userId, productId, quantity) {
    const key = `${CART_PREFIX}${userId}`;
    if (quantity <= 0) {
      await redis.hdel(key, productId);
    } else {
      await redis.hset(key, productId, quantity);
    }
    return this.getCart(userId);
  }

  // 清空购物车
  async clearCart(userId) {
    await redis.del(`${CART_PREFIX}${userId}`);
  }
}

// ========== Stripe 支付 ==========
// services/stripe.js
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const db = require('../db/knex');

class CheckoutService {
  // 创建支付 Intent
  async createPaymentIntent(userId) {
    const cart = await cartService.getCart(userId);
    if (cart.length === 0) throw new Error('购物车为空');

    // 计算总价 (从数据库取最新价格)
    let total = 0;
    for (const item of cart) {
      const product = await db('products').where({ id: item.productId }).first();
      if (!product || product.stock < item.quantity) {
        throw new Error(`商品 ${product?.name || item.productId} 库存不足`);
      }
      total += parseFloat(product.price) * item.quantity;
    }

    // 创建 Stripe PaymentIntent
    const paymentIntent = await stripe.paymentIntents.create({
      amount: Math.round(total * 100), // 分
      currency: 'usd',
      metadata: { userId: String(userId) },
      automatic_payment_methods: { enabled: true },
    });

    return {
      clientSecret: paymentIntent.client_secret,
      total,
      paymentIntentId: paymentIntent.id,
    };
  }

  // 确认订单 (Webhook 调用)
  async confirmOrder(paymentIntentId) {
    const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);
    const userId = parseInt(paymentIntent.metadata.userId);

    // 数据库事务: 创建订单 + 扣库存
    const order = await db.transaction(async (trx) => {
      const cart = await cartService.getCart(userId);

      // 扣库存
      for (const item of cart) {
        const updated = await trx('products')
          .where({ id: item.productId })
          .where('stock', '>=', item.quantity)
          .decrement('stock', item.quantity);

        if (updated === 0) {
          throw new Error(`商品 ${item.productId} 库存不足`);
        }
      }

      // 创建订单
      const [order] = await trx('orders')
        .insert({
          user_id: userId,
          stripe_payment_intent_id: paymentIntentId,
          status: 'paid',
          total: paymentIntent.amount / 100,
        })
        .returning('*');

      // 创建订单项
      const orderItems = cart.map(item => ({
        order_id: order.id,
        product_id: item.productId,
        quantity: item.quantity,
        price: item.price,
      }));
      await trx('order_items').insert(orderItems);

      // 清空购物车
      await cartService.clearCart(userId);

      return order;
    });

    return order;
  }
}
```


## Webhook 与路由


```
// ========== Stripe Webhook ==========
// routes/webhook.js
const express = require('express');
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

const router = express.Router();

// Stripe Webhook — 接收支付事件
router.post('/stripe', express.raw({ type: 'application/json' }), async (req, res) => {
  const sig = req.headers['stripe-signature'];

  let event;
  try {
    event = stripe.webhooks.constructEvent(
      req.body,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET
    );
  } catch (err) {
    console.error('Webhook 签名验证失败:', err.message);
    return res.status(400).send('Webhook signature verification failed');
  }

  // 异步处理事件
  switch (event.type) {
    case 'payment_intent.succeeded':
      await checkoutService.confirmOrder(event.data.object.id);
      break;

    case 'payment_intent.payment_failed':
      console.log('支付失败:', event.data.object.id);
      break;
  }

  res.json({ received: true });
});

// ========== 结算路由 ==========
// routes/checkout.js
const router = require('express').Router();
const checkoutService = new CheckoutService();

// 创建支付
router.post('/create-payment', authenticate, async (req, res) => {
  try {
    const result = await checkoutService.createPaymentIntent(req.userId);
    res.json({ clientSecret: result.clientSecret, total: result.total });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// ========== Docker Compose ==========
// docker-compose.yml:
// version: '3.8'
// services:
//   api:
//     build: .
//     ports: ["3000:3000"]
//     environment:
//       STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY}
//       STRIPE_WEBHOOK_SECRET: ${STRIPE_WEBHOOK_SECRET}
//     depends_on:
//       db: { condition: service_healthy }
//       redis: { condition: service_started }
//
//   db:
//     image: postgres:16-alpine
//     environment:
//       POSTGRES_PASSWORD: secret
//     healthcheck:
//       test: pg_isready -U postgres
//
//   redis:
//     image: redis:7-alpine
```


> **Note:** 💡 Express + Stripe 电商要点: Redis 购物车 (Hash + TTL); Stripe PaymentIntent 支付流程; Webhook 事件驱动确认订单; PostgreSQL 事务扣库存 (decrement + stock条件); 幂等 Webhook 处理; clientSecret 前端安全; 签名验证 stripe-signature。


## 练习


<!-- Converted from: 5_项目实战 Express  Stripe 电商.html -->
