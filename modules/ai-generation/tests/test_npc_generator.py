import os
import sys
import json
import pytest
import numpy as np
import tensorflow as tf

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npc.npc_generator import NPCPersonalityModel

@pytest.fixture
def npc_generator():
    """Create an NPC generator instance"""
    return NPCPersonalityModel()

def test_npc_generator_initialization(npc_generator):
    """Test NPC generator initialization"""
    assert hasattr(npc_generator, 'personality_model')
    assert hasattr(npc_generator, 'emotion_model')
    assert hasattr(npc_generator, 'dialogue_model')
    
    # Check model types
    assert isinstance(npc_generator.personality_model, tf.keras.Model)
    assert isinstance(npc_generator.emotion_model, tf.keras.Model)
    assert isinstance(npc_generator.dialogue_model, tf.keras.Model)

def test_npc_personality_generation(npc_generator):
    """Test NPC personality generation"""
    npc_profile = npc_generator.generate_npc_personality()
    
    # Validate profile structure
    assert 'id' in npc_profile
    assert 'name' in npc_profile
    assert 'age' in npc_profile
    assert 'personality_traits' in npc_profile
    assert 'background' in npc_profile
    assert 'occupation' in npc_profile
    assert 'skills' in npc_profile
    assert 'dialogue_style' in npc_profile
    assert 'initial_mood' in npc_profile
    
    # Validate personality traits
    personality_traits = npc_profile['personality_traits']
    assert len(personality_traits) == 5
    assert all(0 <= trait <= 1 for trait in personality_traits.values())
    assert set(personality_traits.keys()) == set(NPCPersonalityModel.PERSONALITY_TRAITS)

def test_background_generation(npc_generator):
    """Test NPC background generation"""
    npc_profile = npc_generator.generate_npc_personality()
    background = npc_profile['background']
    
    assert 'origin' in background
    assert 'significant_events' in background
    assert isinstance(background['significant_events'], list)
    assert 1 <= len(background['significant_events']) <= 3

def test_occupation_and_skills(npc_generator):
    """Test NPC occupation and skills generation"""
    npc_profile = npc_generator.generate_npc_personality()
    
    assert 'occupation' in npc_profile
    assert 'skills' in npc_profile
    assert isinstance(npc_profile['skills'], list)
    assert len(npc_profile['skills']) > 0

def test_name_generation(npc_generator):
    """Test NPC name generation"""
    npc_profile = npc_generator.generate_npc_personality()
    name = npc_profile['name']
    
    assert isinstance(name, str)
    assert len(name.split()) == 2  # First and last name
    assert all(part.strip() for part in name.split())

def test_dialogue_style_generation(npc_generator):
    """Test NPC dialogue style generation"""
    npc_profile = npc_generator.generate_npc_personality()
    dialogue_style = npc_profile['dialogue_style']
    
    assert isinstance(dialogue_style, dict)
    assert all(isinstance(value, bool) for value in dialogue_style.values())
    assert set(dialogue_style.keys()).issubset({
        'verbose', 'direct', 'emotional', 'sarcastic'
    })

def test_dialogue_generation(npc_generator):
    """Test NPC dialogue generation"""
    npc_profile = npc_generator.generate_npc_personality()
    
    # Generate dialogue
    dialogue = npc_generator.generate_dialogue(npc_profile)
    
    assert isinstance(dialogue, str)
    assert len(dialogue) > 0

def test_npc_profile_saving(npc_generator, tmp_path):
    """Test NPC profile saving mechanism"""
    # Temporarily override save directory
    original_save_method = npc_generator._save_npc_profile
    
    def mock_save_method(npc_profile):
        save_path = tmp_path / f'npc_profile_{npc_profile["id"]}.json'
        with open(save_path, 'w') as f:
            json.dump(npc_profile, f, indent=2)
    
    npc_generator._save_npc_profile = mock_save_method
    
    # Generate and save NPC profile
    npc_profile = npc_generator.generate_npc_personality()
    
    # Check saved file
    saved_file = tmp_path / f'npc_profile_{npc_profile["id"]}.json'
    assert saved_file.exists()
    
    # Restore original save method
    npc_generator._save_npc_profile = original_save_method

def test_reproducibility(npc_generator):
    """Test NPC generation reproducibility with fixed seed"""
    # Set fixed random seed
    np.random.seed(42)
    
    # Generate multiple NPCs
    npcs = [npc_generator.generate_npc_personality() for _ in range(3)]
    
    # Check that generated NPCs have unique IDs
    npc_ids = [npc['id'] for npc in npcs]
    assert len(set(npc_ids)) == 3

def test_emotion_mapping(npc_generator):
    """Test NPC emotion mapping"""
    npc_profile = npc_generator.generate_npc_personality()
    initial_mood = npc_profile['initial_mood']
    
    assert initial_mood in NPCPersonalityModel.EMOTIONS

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
