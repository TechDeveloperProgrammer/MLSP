import os
import uuid
import logging
import sqlite3
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib

class EconomySystem:
    """
    Advanced Virtual Economy System for Minecraft Servers
    
    Features:
    - Multi-currency support
    - Transaction logging
    - Economic events and achievements
    - Anti-fraud mechanisms
    - Player economic profiles
    """
    
    @dataclass
    class Currency:
        """Represents a virtual currency"""
        id: str
        name: str
        symbol: str
        exchange_rate: float
        is_primary: bool = False
    
    @dataclass
    class Transaction:
        """Detailed transaction record"""
        id: str
        sender_uuid: str
        recipient_uuid: str
        amount: float
        currency_id: str
        timestamp: float
        transaction_type: str
        metadata: Optional[Dict[str, Any]] = None
    
    @dataclass
    class PlayerEconomicProfile:
        """Comprehensive economic profile for a player"""
        player_uuid: str
        balances: Dict[str, float]
        total_earned: Dict[str, float]
        total_spent: Dict[str, float]
        achievements: List[str]
        last_login: float
        playtime_economic_bonus: float
    
    def __init__(self, data_dir: str = '/opt/mlsp/economy'):
        """
        Initialize the Economy System
        
        :param data_dir: Directory to store economic data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger('MLSPEconomySystem')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Database paths
        self.currencies_db = os.path.join(data_dir, 'currencies.sqlite')
        self.transactions_db = os.path.join(data_dir, 'transactions.sqlite')
        self.players_db = os.path.join(data_dir, 'players.sqlite')
        
        # Initialize databases
        self._initialize_databases()
        
        # Default currencies
        self._setup_default_currencies()
    
    def _initialize_databases(self):
        """Initialize SQLite databases for economic system"""
        try:
            # Currencies database
            with sqlite3.connect(self.currencies_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS currencies (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        exchange_rate REAL NOT NULL,
                        is_primary INTEGER NOT NULL
                    )
                ''')
                conn.commit()
            
            # Transactions database
            with sqlite3.connect(self.transactions_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id TEXT PRIMARY KEY,
                        sender_uuid TEXT NOT NULL,
                        recipient_uuid TEXT NOT NULL,
                        amount REAL NOT NULL,
                        currency_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        transaction_type TEXT NOT NULL,
                        metadata TEXT
                    )
                ''')
                conn.commit()
            
            # Players economic profile database
            with sqlite3.connect(self.players_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS player_profiles (
                        player_uuid TEXT PRIMARY KEY,
                        balances TEXT NOT NULL,
                        total_earned TEXT NOT NULL,
                        total_spent TEXT NOT NULL,
                        achievements TEXT NOT NULL,
                        last_login REAL NOT NULL,
                        playtime_economic_bonus REAL NOT NULL
                    )
                ''')
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def _setup_default_currencies(self):
        """Set up default currencies for the economy system"""
        default_currencies = [
            self.Currency(
                id='emerald',
                name='Emerald Coin',
                symbol='💎',
                exchange_rate=1.0,
                is_primary=True
            ),
            self.Currency(
                id='gold',
                name='Gold Nugget',
                symbol='🥇',
                exchange_rate=0.5,
                is_primary=False
            )
        ]
        
        for currency in default_currencies:
            self.create_or_update_currency(currency)
    
    def create_or_update_currency(self, currency: Currency):
        """
        Create or update a currency in the system
        
        :param currency: Currency to create or update
        """
        try:
            with sqlite3.connect(self.currencies_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO currencies 
                    (id, name, symbol, exchange_rate, is_primary) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    currency.id,
                    currency.name,
                    currency.symbol,
                    currency.exchange_rate,
                    int(currency.is_primary)
                ))
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Currency creation error: {e}")
    
    def create_transaction(
        self, 
        sender_uuid: str, 
        recipient_uuid: str, 
        amount: float, 
        currency_id: str,
        transaction_type: str = 'transfer',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Transaction:
        """
        Create a new transaction
        
        :param sender_uuid: UUID of the sender
        :param recipient_uuid: UUID of the recipient
        :param amount: Transaction amount
        :param currency_id: Currency of the transaction
        :param transaction_type: Type of transaction
        :param metadata: Additional transaction metadata
        :return: Created transaction
        """
        try:
            # Generate unique transaction ID
            transaction_id = str(uuid.uuid4())
            
            # Create transaction object
            transaction = self.Transaction(
                id=transaction_id,
                sender_uuid=sender_uuid,
                recipient_uuid=recipient_uuid,
                amount=amount,
                currency_id=currency_id,
                timestamp=datetime.now().timestamp(),
                transaction_type=transaction_type,
                metadata=metadata
            )
            
            # Store transaction
            with sqlite3.connect(self.transactions_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO transactions 
                    (id, sender_uuid, recipient_uuid, amount, currency_id, 
                    timestamp, transaction_type, metadata) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction.id,
                    transaction.sender_uuid,
                    transaction.recipient_uuid,
                    transaction.amount,
                    transaction.currency_id,
                    transaction.timestamp,
                    transaction.transaction_type,
                    json.dumps(transaction.metadata) if transaction.metadata else None
                ))
                conn.commit()
            
            return transaction
        
        except sqlite3.Error as e:
            self.logger.error(f"Transaction creation error: {e}")
            raise
    
    def create_player_profile(self, player_uuid: str) -> PlayerEconomicProfile:
        """
        Create a new player economic profile
        
        :param player_uuid: UUID of the player
        :return: Created player economic profile
        """
        try:
            # Default profile
            profile = self.PlayerEconomicProfile(
                player_uuid=player_uuid,
                balances={'emerald': 100.0, 'gold': 50.0},
                total_earned={'emerald': 0.0, 'gold': 0.0},
                total_spent={'emerald': 0.0, 'gold': 0.0},
                achievements=[],
                last_login=datetime.now().timestamp(),
                playtime_economic_bonus=1.0
            )
            
            # Store profile
            with sqlite3.connect(self.players_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO player_profiles 
                    (player_uuid, balances, total_earned, total_spent, 
                    achievements, last_login, playtime_economic_bonus) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.player_uuid,
                    json.dumps(profile.balances),
                    json.dumps(profile.total_earned),
                    json.dumps(profile.total_spent),
                    json.dumps(profile.achievements),
                    profile.last_login,
                    profile.playtime_economic_bonus
                ))
                conn.commit()
            
            return profile
        
        except sqlite3.Error as e:
            self.logger.error(f"Player profile creation error: {e}")
            raise
    
    def transfer_currency(
        self, 
        sender_uuid: str, 
        recipient_uuid: str, 
        amount: float, 
        currency_id: str
    ) -> Transaction:
        """
        Transfer currency between players
        
        :param sender_uuid: UUID of the sender
        :param recipient_uuid: UUID of the recipient
        :param amount: Amount to transfer
        :param currency_id: Currency to transfer
        :return: Created transaction
        """
        try:
            # Validate transaction
            sender_profile = self.get_player_profile(sender_uuid)
            recipient_profile = self.get_player_profile(recipient_uuid)
            
            # Check sender balance
            if sender_profile.balances.get(currency_id, 0) < amount:
                raise ValueError("Insufficient balance")
            
            # Update balances
            sender_profile.balances[currency_id] -= amount
            recipient_profile.balances[currency_id] += amount
            
            # Update player profiles
            self._update_player_profile(sender_profile)
            self._update_player_profile(recipient_profile)
            
            # Create transaction
            transaction = self.create_transaction(
                sender_uuid=sender_uuid,
                recipient_uuid=recipient_uuid,
                amount=amount,
                currency_id=currency_id,
                transaction_type='player_transfer'
            )
            
            return transaction
        
        except Exception as e:
            self.logger.error(f"Currency transfer error: {e}")
            raise
    
    def get_player_profile(self, player_uuid: str) -> Optional[PlayerEconomicProfile]:
        """
        Retrieve a player's economic profile
        
        :param player_uuid: UUID of the player
        :return: Player's economic profile
        """
        try:
            with sqlite3.connect(self.players_db) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM player_profiles WHERE player_uuid = ?', (player_uuid,))
                row = cursor.fetchone()
                
                if not row:
                    return self.create_player_profile(player_uuid)
                
                return self.PlayerEconomicProfile(
                    player_uuid=row[0],
                    balances=json.loads(row[1]),
                    total_earned=json.loads(row[2]),
                    total_spent=json.loads(row[3]),
                    achievements=json.loads(row[4]),
                    last_login=row[5],
                    playtime_economic_bonus=row[6]
                )
        
        except sqlite3.Error as e:
            self.logger.error(f"Player profile retrieval error: {e}")
            raise
    
    def _update_player_profile(self, profile: PlayerEconomicProfile):
        """
        Update a player's economic profile
        
        :param profile: Updated player economic profile
        """
        try:
            with sqlite3.connect(self.players_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE player_profiles 
                    SET balances = ?, total_earned = ?, total_spent = ?, 
                        achievements = ?, last_login = ?, playtime_economic_bonus = ? 
                    WHERE player_uuid = ?
                ''', (
                    json.dumps(profile.balances),
                    json.dumps(profile.total_earned),
                    json.dumps(profile.total_spent),
                    json.dumps(profile.achievements),
                    profile.last_login,
                    profile.playtime_economic_bonus,
                    profile.player_uuid
                ))
                conn.commit()
        
        except sqlite3.Error as e:
            self.logger.error(f"Player profile update error: {e}")
            raise

def main():
    """Example usage of the Economy System"""
    economy = EconomySystem()
    
    try:
        # Create player UUIDs
        player1_uuid = str(uuid.uuid4())
        player2_uuid = str(uuid.uuid4())
        
        # Create player profiles
        player1_profile = economy.create_player_profile(player1_uuid)
        player2_profile = economy.create_player_profile(player2_uuid)
        
        print("Player 1 Initial Balance:", player1_profile.balances)
        print("Player 2 Initial Balance:", player2_profile.balances)
        
        # Transfer currency
        transaction = economy.transfer_currency(
            sender_uuid=player1_uuid,
            recipient_uuid=player2_uuid,
            amount=50.0,
            currency_id='emerald'
        )
        
        # Retrieve updated profiles
        player1_profile = economy.get_player_profile(player1_uuid)
        player2_profile = economy.get_player_profile(player2_uuid)
        
        print("Player 1 Balance After Transfer:", player1_profile.balances)
        print("Player 2 Balance After Transfer:", player2_profile.balances)
        print("Transaction Details:", asdict(transaction))
    
    except Exception as e:
        print(f"Economy system error: {e}")

if __name__ == '__main__':
    main()
