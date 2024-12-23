import os
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional
import json
import logging
import uuid
import spacy

class NPCPersonalityModel:
    """
    Advanced AI-Driven NPC Personality Generation System
    
    Features:
    - Multi-dimensional personality modeling
    - Contextual behavior generation
    - Emotional intelligence
    - Dynamic interaction capabilities
    """
    
    # Personality Trait Dimensions
    PERSONALITY_TRAITS = [
        'openness', 'conscientiousness', 'extraversion', 
        'agreeableness', 'neuroticism'
    ]
    
    # Emotion Categories
    EMOTIONS = [
        'joy', 'sadness', 'anger', 'fear', 
        'surprise', 'disgust', 'neutral'
    ]
    
    def __init__(
        self, 
        models_dir: str = '/opt/mlsp/npc_models',
        cache_dir: str = '/tmp/mlsp_npc_cache'
    ):
        """
        Initialize NPC Personality Generation System
        
        :param models_dir: Directory for storing ML models
        :param cache_dir: Temporary directory for NPC data
        """
        # Create necessary directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configuration
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        
        # Logging setup
        self.logger = logging.getLogger('NPCPersonalityModel')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Natural Language Processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning("SpaCy English model not found. Some NLP features will be limited.")
            self.nlp = None
        
        # Load or create ML models
        self._load_models()
    
    def _load_models(self):
        """
        Load or create machine learning models for NPC generation
        """
        try:
            # Personality Trait Model
            personality_model_path = os.path.join(self.models_dir, 'personality_model.h5')
            if os.path.exists(personality_model_path):
                self.personality_model = tf.keras.models.load_model(personality_model_path)
            else:
                self.personality_model = self._create_personality_model()
            
            # Emotion Recognition Model
            emotion_model_path = os.path.join(self.models_dir, 'emotion_model.h5')
            if os.path.exists(emotion_model_path):
                self.emotion_model = tf.keras.models.load_model(emotion_model_path)
            else:
                self.emotion_model = self._create_emotion_model()
            
            # Dialogue Generation Model
            dialogue_model_path = os.path.join(self.models_dir, 'dialogue_model.h5')
            if os.path.exists(dialogue_model_path):
                self.dialogue_model = tf.keras.models.load_model(dialogue_model_path)
            else:
                self.dialogue_model = self._create_dialogue_model()
        
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            raise
    
    def _create_personality_model(self) -> tf.keras.Model:
        """
        Create a neural network for personality trait generation
        
        :return: Keras personality model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.PERSONALITY_TRAITS), activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_emotion_model(self) -> tf.keras.Model:
        """
        Create a neural network for emotion recognition
        
        :return: Keras emotion model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.EMOTIONS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_dialogue_model(self) -> tf.keras.Model:
        """
        Create a dialogue generation model using sequence-to-sequence architecture
        
        :return: Keras dialogue generation model
        """
        encoder_inputs = tf.keras.layers.Input(shape=(None, 100))
        encoder = tf.keras.layers.LSTM(256, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = tf.keras.layers.Input(shape=(None, 100))
        decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, 
            initial_state=encoder_states
        )
        
        decoder_dense = tf.keras.layers.Dense(100, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_npc_personality(self) -> Dict[str, Any]:
        """
        Generate a comprehensive NPC personality profile
        
        :return: NPC personality metadata
        """
        # Generate unique NPC ID
        npc_id = str(uuid.uuid4())
        
        # Personality trait generation
        personality_input = np.random.rand(1, 10)
        personality_traits = self.personality_model.predict(personality_input)[0]
        
        # Background generation
        background = self._generate_background()
        
        # Occupation and skills
        occupation, skills = self._generate_occupation_and_skills()
        
        # Dialogue style
        dialogue_style = self._generate_dialogue_style(personality_traits)
        
        # NPC personality profile
        npc_profile = {
            'id': npc_id,
            'name': self._generate_name(),
            'age': random.randint(18, 80),
            'personality_traits': dict(zip(self.PERSONALITY_TRAITS, personality_traits)),
            'background': background,
            'occupation': occupation,
            'skills': skills,
            'dialogue_style': dialogue_style,
            'initial_mood': random.choice(self.EMOTIONS)
        }
        
        # Save NPC profile
        self._save_npc_profile(npc_profile)
        
        return npc_profile
    
    def _generate_background(self) -> Dict[str, Any]:
        """
        Generate a contextual background for the NPC
        
        :return: NPC background metadata
        """
        background_templates = [
            "Grew up in a small village",
            "Traveled extensively as a merchant",
            "Trained as a scholar",
            "Worked as a craftsman",
            "Lived through challenging times"
        ]
        
        return {
            'origin': random.choice(background_templates),
            'significant_events': [
                f"Event {i+1}: {random.choice(background_templates)}" 
                for i in range(random.randint(1, 3))
            ]
        }
    
    def _generate_occupation_and_skills(self) -> Tuple[str, List[str]]:
        """
        Generate NPC occupation and associated skills
        
        :return: Tuple of occupation and skills
        """
        occupations = [
            ('Blacksmith', ['metalworking', 'weapon crafting', 'repair']),
            ('Merchant', ['trading', 'negotiation', 'resource management']),
            ('Scholar', ['research', 'ancient knowledge', 'magic theory']),
            ('Adventurer', ['combat', 'survival', 'exploration']),
            ('Farmer', ['agriculture', 'animal husbandry', 'resource gathering'])
        ]
        
        occupation, skills = random.choice(occupations)
        
        return occupation, skills
    
    def _generate_name(self) -> str:
        """
        Generate a contextually appropriate name
        
        :return: Generated NPC name
        """
        first_names = [
            'Aria', 'Rowan', 'Kai', 'Lyra', 'Ezra', 
            'Nova', 'Sage', 'Finn', 'Zara', 'Orion'
        ]
        last_names = [
            'Stormwind', 'Ironheart', 'Silverforge', 'Moonwhisper', 
            'Brightstone', 'Shadowmere', 'Oakenheart'
        ]
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_dialogue_style(
        self, 
        personality_traits: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate a unique dialogue style based on personality
        
        :param personality_traits: NPC personality traits
        :return: Dialogue style metadata
        """
        dialogue_styles = {
            'verbose': personality_traits[0] > 0.7,
            'direct': personality_traits[1] > 0.6,
            'emotional': personality_traits[4] > 0.5,
            'sarcastic': personality_traits[0] > 0.8 and personality_traits[2] > 0.6
        }
        
        return {
            style: value 
            for style, value in dialogue_styles.items()
        }
    
    def _save_npc_profile(self, npc_profile: Dict[str, Any]):
        """
        Save generated NPC profile
        
        :param npc_profile: NPC personality profile
        """
        try:
            profile_path = os.path.join(
                self.cache_dir, 
                f'npc_profile_{npc_profile["id"]}.json'
            )
            
            with open(profile_path, 'w') as f:
                json.dump(npc_profile, f, indent=2)
            
            self.logger.info(f"NPC profile saved: {npc_profile['name']}")
        
        except Exception as e:
            self.logger.error(f"NPC profile save error: {e}")
    
    def generate_dialogue(
        self, 
        npc_profile: Dict[str, Any], 
        context: Optional[str] = None
    ) -> str:
        """
        Generate contextual dialogue for NPC
        
        :param npc_profile: NPC personality profile
        :param context: Dialogue context
        :return: Generated dialogue
        """
        try:
            # Placeholder dialogue generation
            # Future implementation will use more advanced NLP and ML techniques
            dialogue_templates = [
                "Greetings, traveler! What brings you to these parts?",
                "I've been waiting for someone like you to arrive.",
                "The road is long and treacherous. Be careful.",
                "I have a tale that might interest you..."
            ]
            
            return random.choice(dialogue_templates)
        
        except Exception as e:
            self.logger.error(f"Dialogue generation error: {e}")
            return "Hello there."

def main():
    """Example usage of NPC Personality Generator"""
    npc_generator = NPCPersonalityModel()
    
    # Generate multiple NPCs
    for _ in range(5):
        npc_profile = npc_generator.generate_npc_personality()
        print("\nNPC Profile:")
        print(json.dumps(npc_profile, indent=2))
        
        # Generate dialogue
        dialogue = npc_generator.generate_dialogue(npc_profile)
        print(f"\nDialogue: {dialogue}")

if __name__ == '__main__':
    main()
