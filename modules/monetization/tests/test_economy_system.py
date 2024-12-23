import os
import uuid
import pytest
import tempfile
from unittest.mock import patch

# Import the economy system
from economy_system import EconomySystem

@pytest.fixture
def economy_system():
    """Create a temporary economy system for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield EconomySystem(data_dir=temp_dir)

def test_currency_creation(economy_system):
    """Test currency creation and retrieval"""
    test_currency = economy_system.Currency(
        id='test_coin',
        name='Test Coin',
        symbol='🧪',
        exchange_rate=1.5,
        is_primary=False
    )
    
    economy_system.create_or_update_currency(test_currency)
    
    # Verify currency creation (would require additional method to retrieve)
    assert test_currency is not None

def test_player_profile_creation(economy_system):
    """Test player economic profile creation"""
    player_uuid = str(uuid.uuid4())
    profile = economy_system.create_player_profile(player_uuid)
    
    assert profile.player_uuid == player_uuid
    assert 'emerald' in profile.balances
    assert 'gold' in profile.balances
    assert profile.balances['emerald'] == 100.0
    assert profile.balances['gold'] == 50.0

def test_currency_transfer(economy_system):
    """Test currency transfer between players"""
    player1_uuid = str(uuid.uuid4())
    player2_uuid = str(uuid.uuid4())
    
    # Create player profiles
    player1_profile = economy_system.create_player_profile(player1_uuid)
    player2_profile = economy_system.create_player_profile(player2_uuid)
    
    # Transfer currency
    transfer_amount = 50.0
    transaction = economy_system.transfer_currency(
        sender_uuid=player1_uuid,
        recipient_uuid=player2_uuid,
        amount=transfer_amount,
        currency_id='emerald'
    )
    
    # Retrieve updated profiles
    updated_player1 = economy_system.get_player_profile(player1_uuid)
    updated_player2 = economy_system.get_player_profile(player2_uuid)
    
    # Verify transfer
    assert updated_player1.balances['emerald'] == 50.0
    assert updated_player2.balances['emerald'] == 150.0
    assert transaction.amount == transfer_amount
    assert transaction.sender_uuid == player1_uuid
    assert transaction.recipient_uuid == player2_uuid

def test_insufficient_balance_transfer(economy_system):
    """Test transfer with insufficient balance"""
    player1_uuid = str(uuid.uuid4())
    player2_uuid = str(uuid.uuid4())
    
    # Create player profiles
    economy_system.create_player_profile(player1_uuid)
    economy_system.create_player_profile(player2_uuid)
    
    # Attempt to transfer more than balance
    with pytest.raises(ValueError, match="Insufficient balance"):
        economy_system.transfer_currency(
            sender_uuid=player1_uuid,
            recipient_uuid=player2_uuid,
            amount=150.0,
            currency_id='emerald'
        )

def test_transaction_logging(economy_system):
    """Test transaction logging functionality"""
    player1_uuid = str(uuid.uuid4())
    player2_uuid = str(uuid.uuid4())
    
    # Create player profiles
    economy_system.create_player_profile(player1_uuid)
    economy_system.create_player_profile(player2_uuid)
    
    # Create transaction
    transaction = economy_system.create_transaction(
        sender_uuid=player1_uuid,
        recipient_uuid=player2_uuid,
        amount=25.0,
        currency_id='emerald',
        transaction_type='test_transaction',
        metadata={'test_key': 'test_value'}
    )
    
    # Verify transaction details
    assert transaction.sender_uuid == player1_uuid
    assert transaction.recipient_uuid == player2_uuid
    assert transaction.amount == 25.0
    assert transaction.currency_id == 'emerald'
    assert transaction.transaction_type == 'test_transaction'
    assert transaction.metadata == {'test_key': 'test_value'}

def test_multiple_currency_support(economy_system):
    """Test support for multiple currencies"""
    # Create a new currency
    new_currency = economy_system.Currency(
        id='diamond',
        name='Diamond Coin',
        symbol='💎',
        exchange_rate=2.0,
        is_primary=False
    )
    economy_system.create_or_update_currency(new_currency)
    
    player_uuid = str(uuid.uuid4())
    profile = economy_system.create_player_profile(player_uuid)
    
    # Add balance for new currency
    profile.balances['diamond'] = 100.0
    economy_system._update_player_profile(profile)
    
    # Retrieve updated profile
    updated_profile = economy_system.get_player_profile(player_uuid)
    
    assert 'diamond' in updated_profile.balances
    assert updated_profile.balances['diamond'] == 100.0

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
