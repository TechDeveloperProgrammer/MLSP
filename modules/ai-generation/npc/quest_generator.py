import os
import random
import uuid
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Tuple

class QuestGenerator:
    """
    Advanced AI-Driven Quest Generation System
    
    Features:
    - Contextually-aware quest generation
    - Personality-driven quest design
    - Dynamic difficulty scaling
    - Narrative complexity
    - Player interaction optimization
    """
    
    # Quest Types
    QUEST_TYPES = [
        'exploration', 'rescue', 'collection', 
        'elimination', 'escort', 'crafting', 
        'investigation', 'diplomatic', 'survival'
    ]
    
    # Quest Difficulty Levels
    DIFFICULTY_LEVELS = ['easy', 'medium', 'hard', 'legendary']
    
    # Reward Categories
    REWARD_TYPES = [
        'item', 'currency', 'experience', 
        'reputation', 'skill_upgrade', 'unique_ability'
    ]
    
    def __init__(
        self, 
        models_dir: str = '/opt/mlsp/quest_models',
        cache_dir: str = '/tmp/mlsp_quest_cache'
    ):
        """
        Initialize Quest Generation System
        
        :param models_dir: Directory for storing ML models
        :param cache_dir: Temporary directory for quest data
        """
        # Create necessary directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configuration
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        
        # Logging setup
        self.logger = logging.getLogger('QuestGenerator')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Load or create ML models
        self._load_models()
    
    def _load_models(self):
        """
        Load or create machine learning models for quest generation
        """
        try:
            # Quest Generation Model
            quest_model_path = os.path.join(self.models_dir, 'quest_generation_model.h5')
            if os.path.exists(quest_model_path):
                self.quest_model = tf.keras.models.load_model(quest_model_path)
            else:
                self.quest_model = self._create_quest_generation_model()
            
            # Quest Difficulty Prediction Model
            difficulty_model_path = os.path.join(self.models_dir, 'quest_difficulty_model.h5')
            if os.path.exists(difficulty_model_path):
                self.difficulty_model = tf.keras.models.load_model(difficulty_model_path)
            else:
                self.difficulty_model = self._create_difficulty_prediction_model()
        
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise
    
    def _create_quest_generation_model(self) -> tf.keras.Model:
        """
        Create a neural network for quest generation
        
        :return: Keras quest generation model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.QUEST_TYPES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_difficulty_prediction_model(self) -> tf.keras.Model:
        """
        Create a neural network for quest difficulty prediction
        
        :return: Keras difficulty prediction model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.DIFFICULTY_LEVELS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_quest(
        self, 
        npc_profile: Dict[str, Any], 
        world_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a contextually-aware quest
        
        :param npc_profile: NPC personality profile
        :param world_context: Contextual world information
        :return: Generated quest metadata
        """
        # Generate unique quest ID
        quest_id = str(uuid.uuid4())
        
        # Determine quest type based on NPC personality
        quest_type = self._determine_quest_type(npc_profile)
        
        # Generate quest narrative
        narrative = self._generate_quest_narrative(
            quest_type, npc_profile, world_context
        )
        
        # Determine quest difficulty
        difficulty = self._predict_quest_difficulty(
            npc_profile, world_context
        )
        
        # Generate quest objectives
        objectives = self._generate_quest_objectives(
            quest_type, difficulty
        )
        
        # Generate quest rewards
        rewards = self._generate_quest_rewards(difficulty)
        
        # Construct quest metadata
        quest = {
            'id': quest_id,
            'title': narrative['title'],
            'description': narrative['description'],
            'type': quest_type,
            'difficulty': difficulty,
            'npc_giver': {
                'name': npc_profile['name'],
                'id': npc_profile['id']
            },
            'objectives': objectives,
            'rewards': rewards,
            'narrative_context': narrative
        }
        
        # Save quest
        self._save_quest(quest)
        
        return quest
    
    def _determine_quest_type(
        self, 
        npc_profile: Dict[str, Any]
    ) -> str:
        """
        Determine quest type based on NPC personality
        
        :param npc_profile: NPC personality profile
        :return: Selected quest type
        """
        personality_traits = npc_profile['personality_traits']
        
        # Personality-driven quest type selection
        quest_type_mapping = {
            'openness': ['exploration', 'investigation'],
            'conscientiousness': ['collection', 'crafting'],
            'extraversion': ['diplomatic', 'escort'],
            'agreeableness': ['rescue', 'survival'],
            'neuroticism': ['elimination']
        }
        
        # Find dominant personality trait
        dominant_trait = max(
            personality_traits, 
            key=personality_traits.get
        )
        
        return random.choice(
            quest_type_mapping.get(dominant_trait, self.QUEST_TYPES)
        )
    
    def _generate_quest_narrative(
        self, 
        quest_type: str, 
        npc_profile: Dict[str, Any],
        world_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate a compelling quest narrative
        
        :param quest_type: Type of quest
        :param npc_profile: NPC personality profile
        :param world_context: Contextual world information
        :return: Quest narrative metadata
        """
        narrative_templates = {
            'exploration': [
                {
                    'title': "Uncharted Horizons",
                    'description': "Explore the mysterious {location} and uncover its hidden secrets."
                }
            ],
            'rescue': [
                {
                    'title': "Lost and Desperate",
                    'description': "A {character} is in grave danger and needs your immediate help."
                }
            ],
            # Add more narrative templates for other quest types
        }
        
        # Select narrative template
        template = random.choice(
            narrative_templates.get(quest_type, narrative_templates['exploration'])
        )
        
        # Customize narrative based on NPC personality and world context
        narrative = {
            'title': template['title'],
            'description': template['description'].format(
                location=world_context.get('biome', 'unknown region') if world_context else 'unknown region',
                character=npc_profile['name']
            )
        }
        
        return narrative
    
    def _predict_quest_difficulty(
        self, 
        npc_profile: Dict[str, Any],
        world_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Predict quest difficulty
        
        :param npc_profile: NPC personality profile
        :param world_context: Contextual world information
        :return: Quest difficulty level
        """
        # Prepare input features
        input_features = np.concatenate([
            list(npc_profile['personality_traits'].values()),
            [random.random() for _ in range(10)]  # Additional randomness
        ])
        
        # Predict difficulty
        difficulty_probs = self.difficulty_model.predict(
            input_features.reshape(1, -1)
        )[0]
        
        # Select difficulty based on probabilities
        return random.choices(
            self.DIFFICULTY_LEVELS, 
            weights=difficulty_probs
        )[0]
    
    def _generate_quest_objectives(
        self, 
        quest_type: str, 
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """
        Generate quest objectives
        
        :param quest_type: Type of quest
        :param difficulty: Quest difficulty level
        :return: List of quest objectives
        """
        difficulty_multipliers = {
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'legendary': 4
        }
        
        objectives_templates = {
            'exploration': [
                "Discover {count} unique landmarks",
                "Map {count} unexplored regions"
            ],
            'rescue': [
                "Locate and rescue {count} individuals",
                "Protect a group of {count} survivors"
            ],
            # Add more objective templates for other quest types
        }
        
        # Select objectives based on quest type and difficulty
        num_objectives = random.randint(1, difficulty_multipliers[difficulty])
        objectives = []
        
        for _ in range(num_objectives):
            template = random.choice(
                objectives_templates.get(quest_type, objectives_templates['exploration'])
            )
            
            objectives.append({
                'description': template.format(
                    count=random.randint(1, difficulty_multipliers[difficulty] * 3)
                ),
                'completed': False
            })
        
        return objectives
    
    def _generate_quest_rewards(
        self, 
        difficulty: str
    ) -> List[Dict[str, Any]]:
        """
        Generate quest rewards
        
        :param difficulty: Quest difficulty level
        :return: List of quest rewards
        """
        difficulty_multipliers = {
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'legendary': 4
        }
        
        num_rewards = random.randint(1, difficulty_multipliers[difficulty])
        rewards = []
        
        for _ in range(num_rewards):
            reward_type = random.choice(self.REWARD_TYPES)
            rewards.append({
                'type': reward_type,
                'value': random.randint(
                    10 * difficulty_multipliers[difficulty], 
                    100 * difficulty_multipliers[difficulty]
                )
            })
        
        return rewards
    
    def _save_quest(self, quest: Dict[str, Any]):
        """
        Save generated quest
        
        :param quest: Quest metadata
        """
        try:
            quest_path = os.path.join(
                self.cache_dir, 
                f'quest_{quest["id"]}.json'
            )
            
            with open(quest_path, 'w') as f:
                json.dump(quest, f, indent=2)
            
            self.logger.info(f"Quest saved: {quest['title']}")
        
        except Exception as e:
            self.logger.error(f"Quest save error: {e}")

def main():
    """Example usage of Quest Generator"""
    # Simulated NPC profile
    npc_profile = {
        'id': str(uuid.uuid4()),
        'name': 'Aria Stormwind',
        'personality_traits': {
            'openness': 0.8,
            'conscientiousness': 0.6,
            'extraversion': 0.4,
            'agreeableness': 0.7,
            'neuroticism': 0.3
        }
    }
    
    # Simulated world context
    world_context = {
        'biome': 'dense_forest',
        'difficulty': 'medium'
    }
    
    quest_generator = QuestGenerator()
    
    # Generate multiple quests
    for _ in range(3):
        quest = quest_generator.generate_quest(
            npc_profile, 
            world_context
        )
        print("\nGenerated Quest:")
        print(json.dumps(quest, indent=2))

if __name__ == '__main__':
    main()
