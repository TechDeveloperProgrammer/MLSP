import os
import sys
import json
import pytest
import numpy as np
import tensorflow as tf
import uuid

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npc.quest_generator import QuestGenerator

@pytest.fixture
def quest_generator():
    """Create a quest generator instance"""
    return QuestGenerator()

def create_sample_npc_profile():
    """Create a sample NPC profile for testing"""
    return {
        'id': str(uuid.uuid4()),
        'name': 'Test NPC',
        'personality_traits': {
            'openness': 0.7,
            'conscientiousness': 0.6,
            'extraversion': 0.5,
            'agreeableness': 0.4,
            'neuroticism': 0.3
        }
    }

def create_sample_world_context():
    """Create a sample world context for testing"""
    return {
        'biome': 'forest',
        'difficulty': 'medium'
    }

def test_quest_generator_initialization(quest_generator):
    """Test quest generator initialization"""
    assert hasattr(quest_generator, 'quest_model')
    assert hasattr(quest_generator, 'difficulty_model')
    
    # Check model types
    assert isinstance(quest_generator.quest_model, tf.keras.Model)
    assert isinstance(quest_generator.difficulty_model, tf.keras.Model)

def test_quest_generation(quest_generator):
    """Test quest generation"""
    npc_profile = create_sample_npc_profile()
    world_context = create_sample_world_context()
    
    quest = quest_generator.generate_quest(npc_profile, world_context)
    
    # Validate quest structure
    assert 'id' in quest
    assert 'title' in quest
    assert 'description' in quest
    assert 'type' in quest
    assert 'difficulty' in quest
    assert 'npc_giver' in quest
    assert 'objectives' in quest
    assert 'rewards' in quest
    
    # Validate quest type
    assert quest['type'] in QuestGenerator.QUEST_TYPES
    
    # Validate difficulty
    assert quest['difficulty'] in QuestGenerator.DIFFICULTY_LEVELS
    
    # Validate NPC giver
    assert 'name' in quest['npc_giver']
    assert 'id' in quest['npc_giver']
    
    # Validate objectives
    assert isinstance(quest['objectives'], list)
    assert len(quest['objectives']) > 0
    for objective in quest['objectives']:
        assert 'description' in objective
        assert 'completed' in objective
        assert isinstance(objective['completed'], bool)
    
    # Validate rewards
    assert isinstance(quest['rewards'], list)
    assert len(quest['rewards']) > 0
    for reward in quest['rewards']:
        assert 'type' in reward
        assert 'value' in reward
        assert reward['type'] in QuestGenerator.REWARD_TYPES

def test_quest_type_determination(quest_generator):
    """Test quest type determination based on NPC personality"""
    npc_profiles = [
        {
            'personality_traits': {
                'openness': 0.9,
                'conscientiousness': 0.2,
                'extraversion': 0.3,
                'agreeableness': 0.4,
                'neuroticism': 0.1
            }
        },
        {
            'personality_traits': {
                'openness': 0.2,
                'conscientiousness': 0.9,
                'extraversion': 0.3,
                'agreeableness': 0.4,
                'neuroticism': 0.1
            }
        }
    ]
    
    for npc_profile in npc_profiles:
        quest_type = quest_generator._determine_quest_type(npc_profile)
        assert quest_type in QuestGenerator.QUEST_TYPES

def test_quest_difficulty_prediction(quest_generator):
    """Test quest difficulty prediction"""
    npc_profile = create_sample_npc_profile()
    world_context = create_sample_world_context()
    
    difficulty = quest_generator._predict_quest_difficulty(
        npc_profile, world_context
    )
    
    assert difficulty in QuestGenerator.DIFFICULTY_LEVELS

def test_quest_objectives_generation(quest_generator):
    """Test quest objectives generation"""
    quest_types = QuestGenerator.QUEST_TYPES
    difficulty_levels = QuestGenerator.DIFFICULTY_LEVELS
    
    for quest_type in quest_types:
        for difficulty in difficulty_levels:
            objectives = quest_generator._generate_quest_objectives(
                quest_type, difficulty
            )
            
            assert isinstance(objectives, list)
            assert len(objectives) > 0
            
            for objective in objectives:
                assert 'description' in objective
                assert 'completed' in objective
                assert isinstance(objective['completed'], bool)

def test_quest_rewards_generation(quest_generator):
    """Test quest rewards generation"""
    difficulty_levels = QuestGenerator.DIFFICULTY_LEVELS
    
    for difficulty in difficulty_levels:
        rewards = quest_generator._generate_quest_rewards(difficulty)
        
        assert isinstance(rewards, list)
        assert len(rewards) > 0
        
        for reward in rewards:
            assert 'type' in reward
            assert 'value' in reward
            assert reward['type'] in QuestGenerator.REWARD_TYPES

def test_quest_narrative_generation(quest_generator):
    """Test quest narrative generation"""
    npc_profile = create_sample_npc_profile()
    world_context = create_sample_world_context()
    quest_types = QuestGenerator.QUEST_TYPES
    
    for quest_type in quest_types:
        narrative = quest_generator._generate_quest_narrative(
            quest_type, npc_profile, world_context
        )
        
        assert 'title' in narrative
        assert 'description' in narrative
        assert isinstance(narrative['title'], str)
        assert isinstance(narrative['description'], str)

def test_quest_saving(quest_generator, tmp_path):
    """Test quest saving mechanism"""
    # Temporarily override save directory
    original_save_method = quest_generator._save_quest
    
    def mock_save_method(quest):
        save_path = tmp_path / f'quest_{quest["id"]}.json'
        with open(save_path, 'w') as f:
            json.dump(quest, f, indent=2)
    
    quest_generator._save_quest = mock_save_method
    
    # Generate and save quest
    npc_profile = create_sample_npc_profile()
    world_context = create_sample_world_context()
    quest = quest_generator.generate_quest(npc_profile, world_context)
    
    # Check saved file
    saved_file = tmp_path / f'quest_{quest["id"]}.json'
    assert saved_file.exists()
    
    # Restore original save method
    quest_generator._save_quest = original_save_method

def test_reproducibility(quest_generator):
    """Test quest generation reproducibility"""
    npc_profile = create_sample_npc_profile()
    world_context = create_sample_world_context()
    
    # Set fixed random seed
    np.random.seed(42)
    
    # Generate multiple quests
    quests = [
        quest_generator.generate_quest(npc_profile, world_context) 
        for _ in range(3)
    ]
    
    # Check that generated quests have unique IDs
    quest_ids = [quest['id'] for quest in quests]
    assert len(set(quest_ids)) == 3

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
