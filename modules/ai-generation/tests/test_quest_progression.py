import os
import sys
import json
import pytest
import numpy as np
import tensorflow as tf
import uuid
import random

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npc.quest_progression import QuestProgressionSystem

@pytest.fixture
def quest_progression_system():
    """Create a quest progression system instance"""
    return QuestProgressionSystem()

def create_sample_quest():
    """Create a sample quest for testing"""
    return {
        'id': str(uuid.uuid4()),
        'title': 'Test Quest',
        'difficulty_score': 0.5,
        'objectives': [
            {'description': 'Complete objective 1', 'completed': False},
            {'description': 'Complete objective 2', 'completed': False}
        ]
    }

def create_sample_player_data():
    """Create sample player data for testing"""
    return {
        'exploration_progress': random.random(),
        'combat_skill': random.random(),
        'crafting_level': random.random(),
        'trading_experience': random.random(),
        'time_spent': random.randint(100, 3600),
        'progress_percentage': random.random()
    }

def test_quest_progression_initialization(quest_progression_system):
    """Test quest progression system initialization"""
    assert hasattr(quest_progression_system, 'behavior_model')
    assert hasattr(quest_progression_system, 'progression_model')
    
    # Check model types
    assert isinstance(quest_progression_system.behavior_model, tf.keras.Model)
    assert isinstance(quest_progression_system.progression_model, tf.keras.Model)

def test_track_quest_progress(quest_progression_system):
    """Test quest progress tracking"""
    quest = create_sample_quest()
    player_data = create_sample_player_data()
    
    updated_quest = quest_progression_system.track_quest_progress(quest, player_data)
    
    # Validate updated quest structure
    assert 'objectives' in updated_quest
    assert 'state' in updated_quest
    assert 'player_behavior' in updated_quest
    assert 'feedback' in updated_quest
    
    # Validate objectives
    assert len(updated_quest['objectives']) == len(quest['objectives'])
    
    # Validate state
    assert updated_quest['state'] in QuestProgressionSystem.QUEST_STATES
    
    # Validate player behavior
    assert len(updated_quest['player_behavior']) == len(QuestProgressionSystem.INTERACTION_TYPES)
    
    # Validate feedback
    assert 'overall_progress' in updated_quest['feedback']
    assert 'state_description' in updated_quest['feedback']
    assert 'recommendations' in updated_quest['feedback']

def test_predict_player_behavior(quest_progression_system):
    """Test player behavior prediction"""
    player_data = create_sample_player_data()
    
    player_behavior = quest_progression_system._predict_player_behavior(player_data)
    
    # Validate player behavior
    assert len(player_behavior) == len(QuestProgressionSystem.INTERACTION_TYPES)
    assert all(0 <= prob <= 1 for prob in player_behavior.values())
    assert abs(sum(player_behavior.values()) - 1.0) < 1e-6  # Probabilities sum to 1

def test_update_quest_objectives(quest_progression_system):
    """Test quest objectives update"""
    objectives = [
        {'description': 'Objective 1', 'completed': False},
        {'description': 'Objective 2', 'completed': False}
    ]
    
    player_behavior = {
        interaction_type: random.random() 
        for interaction_type in QuestProgressionSystem.INTERACTION_TYPES
    }
    
    updated_objectives = quest_progression_system._update_quest_objectives(
        objectives, player_behavior
    )
    
    # Validate updated objectives
    assert len(updated_objectives) == len(objectives)
    assert all('description' in obj for obj in updated_objectives)
    assert all('completed' in obj for obj in updated_objectives)

def test_predict_quest_progression(quest_progression_system):
    """Test quest progression state prediction"""
    quest = create_sample_quest()
    player_data = create_sample_player_data()
    player_behavior = {
        interaction_type: random.random() 
        for interaction_type in QuestProgressionSystem.INTERACTION_TYPES
    }
    
    progression_state = quest_progression_system._predict_quest_progression(
        quest, player_data, player_behavior
    )
    
    # Validate progression state
    assert progression_state in QuestProgressionSystem.QUEST_STATES

def test_generate_quest_feedback(quest_progression_system):
    """Test quest feedback generation"""
    quest = create_sample_quest()
    objectives = [
        {'description': 'Objective 1', 'completed': True},
        {'description': 'Objective 2', 'completed': False}
    ]
    progression_state = 'in_progress'
    
    feedback = quest_progression_system._generate_quest_feedback(
        quest, objectives, progression_state
    )
    
    # Validate feedback
    assert 'overall_progress' in feedback
    assert 'state_description' in feedback
    assert 'recommendations' in feedback
    
    # Check overall progress calculation
    assert feedback['overall_progress'] == 50.0
    
    # Check state description
    assert feedback['state_description'] == "You're making progress on this quest. Keep going!"

def test_get_state_description(quest_progression_system):
    """Test state description generation"""
    for state in QuestProgressionSystem.QUEST_STATES:
        description = quest_progression_system._get_state_description(state)
        assert isinstance(description, str)
        assert len(description) > 0

def test_generate_quest_recommendations(quest_progression_system):
    """Test quest recommendations generation"""
    quest = create_sample_quest()
    objectives = [
        {'description': 'Objective 1', 'completed': False},
        {'description': 'Objective 2', 'completed': False}
    ]
    
    # Test in_progress state
    in_progress_recommendations = quest_progression_system._generate_quest_recommendations(
        quest, objectives, 'in_progress'
    )
    assert len(in_progress_recommendations) > 0
    
    # Test failed state
    failed_recommendations = quest_progression_system._generate_quest_recommendations(
        quest, objectives, 'failed'
    )
    assert len(failed_recommendations) > 0
    
    # Test on_hold state
    on_hold_recommendations = quest_progression_system._generate_quest_recommendations(
        quest, objectives, 'on_hold'
    )
    assert len(on_hold_recommendations) > 0

def test_quest_progression_saving(quest_progression_system, tmp_path):
    """Test quest progression saving mechanism"""
    # Temporarily override save directory
    original_save_method = quest_progression_system._save_quest_progression
    
    def mock_save_method(quest_progression):
        save_path = tmp_path / f'quest_progression_{quest_progression["id"]}.json'
        with open(save_path, 'w') as f:
            json.dump(quest_progression, f, indent=2)
    
    quest_progression_system._save_quest_progression = mock_save_method
    
    # Generate and save quest progression
    quest = create_sample_quest()
    player_data = create_sample_player_data()
    quest_progression = quest_progression_system.track_quest_progress(quest, player_data)
    
    # Check saved file
    saved_file = tmp_path / f'quest_progression_{quest_progression["id"]}.json'
    assert saved_file.exists()
    
    # Restore original save method
    quest_progression_system._save_quest_progression = original_save_method

def test_reproducibility(quest_progression_system):
    """Test quest progression reproducibility"""
    # Set fixed random seed
    np.random.seed(42)
    
    quest = create_sample_quest()
    player_data = create_sample_player_data()
    
    # Generate multiple quest progressions
    progressions = [
        quest_progression_system.track_quest_progress(quest, player_data) 
        for _ in range(3)
    ]
    
    # Check that generated progressions have unique IDs
    progression_ids = [progression['id'] for progression in progressions]
    assert len(set(progression_ids)) == 3

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
