# 简单区块链实现 (Simple Blockchain)

## 项目需求与功能分析

区块链是一种去中心化的分布式账本技术。本项目用 Python 实现一个简化版区块链，包含区块结构、哈希链、工作量证明和交易验证，帮助理解区块链的核心原理。

### 核心功能

- 区块结构（索引、时间戳、交易、前驱哈希、Nonce）
- SHA-256 哈希链
- 工作量证明 (Proof of Work) 挖矿
- 交易验证
- 区块链有效性验证
- 简单余额查询

## 核心算法原理

### 哈希链

每个区块包含前一个区块的哈希值，形成不可篡改的链。修改任何区块都会导致后续所有区块的哈希失效。

### 工作量证明

挖矿即寻找一个 Nonce，使得区块哈希满足一定条件（前导零）：

```
hash(block_data + nonce) < target
```

难度由前导零的数量决定：难度 d 意味着哈希值的前 d 位必须为 0。

## 完整代码实现

```python
import hashlib
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Transaction:
    sender: str
    receiver: str
    amount: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {'sender': self.sender, 'receiver': self.receiver,
                'amount': self.amount, 'timestamp': self.timestamp}


class Block:
    """区块"""

    def __init__(self, index: int, transactions: List[Transaction],
                 previous_hash: str, timestamp: float = None):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        data = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [t.to_dict() for t in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [t.to_dict() for t in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash,
        }


class Blockchain:
    """区块链"""

    def __init__(self, difficulty: int = 4, mining_reward: float = 10.0):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.mining_reward = mining_reward
        self.target = '0' * difficulty
        # 创建创世区块
        self._create_genesis()

    def _create_genesis(self):
        genesis = Block(0, [], "0" * 64)
        genesis.hash = self._proof_of_work(genesis)
        self.chain.append(genesis)

    def _proof_of_work(self, block: Block) -> str:
        """工作量证明"""
        block.nonce = 0
        h = block.compute_hash()
        while not h.startswith(self.target):
            block.nonce += 1
            h = block.compute_hash()
        return h

    def add_transaction(self, sender: str, receiver: str, amount: float) -> bool:
        """添加交易"""
        if sender != "MINER" and self.get_balance(sender) < amount:
            print(f"余额不足: {sender} 余额 {self.get_balance(sender)}, 需要 {amount}")
            return False
        self.pending_transactions.append(Transaction(sender, receiver, amount))
        return True

    def mine_pending_transactions(self, miner_address: str) -> Optional[Block]:
        """挖矿：打包待处理交易"""
        if not self.pending_transactions:
            print("没有待处理的交易")
            return None

        # 添加挖矿奖励
        self.pending_transactions.append(
            Transaction("MINER", miner_address, self.mining_reward)
        )

        block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            previous_hash=self.chain[-1].hash
        )

        print(f"正在挖矿 (难度={self.difficulty})...")
        start = time.time()
        block.hash = self._proof_of_work(block)
        elapsed = time.time() - start

        self.chain.append(block)
        self.pending_transactions = []

        print(f"  区块 #{block.index} 已挖出!")
        print(f"  Nonce: {block.nonce}")
        print(f"  哈希: {block.hash}")
        print(f"  耗时: {elapsed:.2f}s")
        return block

    def get_balance(self, address: str) -> float:
        """查询余额"""
        balance = 0.0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= tx.amount
                if tx.receiver == address:
                    balance += tx.amount
        return balance

    def is_valid(self) -> bool:
        """验证区块链完整性"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # 验证哈希
            if current.hash != current.compute_hash():
                print(f"区块 #{i} 哈希无效")
                return False

            # 验证链接
            if current.previous_hash != previous.hash:
                print(f"区块 #{i} 链接断裂")
                return False

            # 验证工作量证明
            if not current.hash.startswith(self.target):
                print(f"区块 #{i} 工作量证明无效")
                return False

        return True

    def display(self):
        """显示区块链"""
        print(f"\n{'='*60}")
        print(f"区块链 (长度={len(self.chain)}, 难度={self.difficulty})")
        print(f"{'='*60}")
        for block in self.chain:
            print(f"\n区块 #{block.index}")
            print(f"  时间: {time.ctime(block.timestamp)}")
            print(f"  哈希: {block.hash}")
            print(f"  前驱: {block.previous_hash[:16]}...")
            print(f"  Nonce: {block.nonce}")
            print(f"  交易: {len(block.transactions)} 笔")
            for tx in block.transactions:
                print(f"    {tx.sender} -> {tx.receiver}: {tx.amount}")

    def tamper(self, block_index: int, tx_index: int, new_amount: float):
        """模拟篡改（用于验证检测）"""
        if block_index < len(self.chain):
            block = self.chain[block_index]
            if tx_index < len(block.transactions):
                block.transactions[tx_index].amount = new_amount
                print(f"已篡改区块 #{block.index} 交易 #{tx_index}")


def demo():
    bc = Blockchain(difficulty=4)
    print("=== 简单区块链演示 ===")

    # 添加交易
    bc.add_transaction("Alice", "Bob", 50)
    bc.add_transaction("Bob", "Charlie", 20)
    bc.mine_pending_transactions("Miner1")

    bc.add_transaction("Charlie", "Alice", 10)
    bc.mine_pending_transactions("Miner2")

    # 显示
    bc.display()

    # 余额查询
    print("\n余额查询:")
    for addr in ["Alice", "Bob", "Charlie", "Miner1", "Miner2"]:
        print(f"  {addr}: {bc.get_balance(addr)}")

    # 验证
    print(f"\n区块链有效: {bc.is_valid()}")

    # 篡改测试
    print("\n--- 篡改测试 ---")
    bc.tamper(1, 0, 9999)
    print(f"篡改后区块链有效: {bc.is_valid()}")


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestBlockchain(unittest.TestCase):
    def test_genesis_block(self):
        bc = Blockchain(difficulty=2)
        self.assertEqual(len(bc.chain), 1)
        self.assertEqual(bc.chain[0].index, 0)
        self.assertTrue(bc.chain[0].hash.startswith('00'))

    def test_mine_block(self):
        bc = Blockchain(difficulty=2)
        bc.add_transaction("A", "B", 10)
        block = bc.mine_pending_transactions("Miner")
        self.assertIsNotNone(block)
        self.assertEqual(len(bc.chain), 2)

    def test_balance(self):
        bc = Blockchain(difficulty=2)
        bc.add_transaction("Alice", "Bob", 30)
        bc.mine_pending_transactions("M")
        self.assertEqual(bc.get_balance("Alice"), -30)
        self.assertEqual(bc.get_balance("Bob"), 30)
        self.assertEqual(bc.get_balance("M"), 10)  # 挖矿奖励

    def test_insufficient_balance(self):
        bc = Blockchain(difficulty=2)
        result = bc.add_transaction("Poor", "Rich", 9999)
        self.assertFalse(result)

    def test_chain_valid(self):
        bc = Blockchain(difficulty=2)
        bc.add_transaction("A", "B", 5)
        bc.mine_pending_transactions("M")
        self.assertTrue(bc.is_valid())

    def test_tamper_detected(self):
        bc = Blockchain(difficulty=2)
        bc.add_transaction("A", "B", 5)
        bc.mine_pending_transactions("M")
        bc.chain[1].transactions[0].amount = 9999
        self.assertFalse(bc.is_valid())

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **Merkle 树**：交易哈希树验证
2. **数字签名**：使用非对称加密验证交易身份
3. **P2P 网络**：模拟多节点共识
4. **最长链规则**：处理分叉
5. **智能合约**：简单的合约执行引擎
6. **钱包**：地址生成和余额管理
7. **权益证明 (PoS)**：实现 PoS 共识机制
