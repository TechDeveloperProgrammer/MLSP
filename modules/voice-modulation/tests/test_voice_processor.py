import os
import numpy as np
import pytest
import tempfile
import soundfile as sf
import librosa

# Import the voice processor
from voice_processor import VoiceModulator

@pytest.fixture
def voice_modulator():
    """Create a temporary voice modulator for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield VoiceModulator(
            models_dir=os.path.join(temp_dir, 'models'),
            cache_dir=os.path.join(temp_dir, 'cache')
        )

def generate_test_audio(duration=3, sample_rate=44100):
    """
    Generate a synthetic test audio signal
    
    :param duration: Audio duration in seconds
    :param sample_rate: Sampling rate
    :return: Numpy array of audio data
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.random.normal(size=t.shape)
    return audio

def test_noise_reduction(voice_modulator):
    """Test noise reduction functionality"""
    # Generate noisy audio
    noisy_audio = generate_test_audio()
    
    # Apply noise reduction
    denoised_audio = voice_modulator.reduce_noise(noisy_audio)
    
    # Verify denoising
    assert denoised_audio is not None
    assert len(denoised_audio) == len(noisy_audio)
    assert np.std(denoised_audio) < np.std(noisy_audio)

def test_voice_style_transfer(voice_modulator):
    """Test voice style transfer"""
    # Generate test audio
    original_audio = generate_test_audio()
    
    # Apply style transfer
    robotic_audio = voice_modulator.transfer_voice_style(original_audio, 'robotic')
    deep_audio = voice_modulator.transfer_voice_style(original_audio, 'deep')
    
    # Verify transformations
    assert robotic_audio is not None
    assert deep_audio is not None
    assert not np.array_equal(original_audio, robotic_audio)
    assert not np.array_equal(original_audio, deep_audio)
    assert len(robotic_audio) == len(original_audio)
    assert len(deep_audio) == len(original_audio)

def test_emotion_detection(voice_modulator):
    """Test emotion detection"""
    # Generate test audio
    audio = generate_test_audio()
    
    # Detect emotions
    emotions = voice_modulator.detect_emotion(audio)
    
    # Verify emotion detection
    assert isinstance(emotions, dict)
    assert len(emotions) > 0
    assert all(0 <= prob <= 1 for prob in emotions.values())
    assert set(emotions.keys()) == {'neutral', 'happy', 'sad', 'angry', 'surprised'}

def test_voice_anonymization(voice_modulator):
    """Test voice anonymization"""
    # Generate test audio
    original_audio = generate_test_audio()
    
    # Anonymize voice
    anonymized_audio = voice_modulator.anonymize_voice(original_audio)
    
    # Verify anonymization
    assert anonymized_audio is not None
    assert len(anonymized_audio) == len(original_audio)
    assert not np.array_equal(original_audio, anonymized_audio)

def test_comprehensive_audio_processing(voice_modulator):
    """Test comprehensive audio processing"""
    # Generate test audio
    audio = generate_test_audio()
    
    # Process audio with various configurations
    result = voice_modulator.process_audio(
        audio, 
        config={
            'style_transfer': True,
            'style': 'robotic',
            'anonymize': True
        }
    )
    
    # Verify processing result
    assert 'audio_id' in result
    assert 'processed_audio_path' in result
    assert 'emotions' in result
    assert 'original_duration' in result
    assert 'processed_duration' in result
    
    # Verify processed audio file exists
    assert os.path.exists(result['processed_audio_path'])
    
    # Load processed audio and verify
    processed_audio, _ = sf.read(result['processed_audio_path'])
    assert len(processed_audio) > 0

def test_model_initialization(voice_modulator):
    """Test initialization of ML models"""
    # Verify model attributes
    assert hasattr(voice_modulator, 'noise_reduction_model')
    assert hasattr(voice_modulator, 'style_transfer_model')
    assert hasattr(voice_modulator, 'emotion_model')
    assert hasattr(voice_modulator, 'anonymization_model')

def main():
    """Run all tests"""
    pytest.main([__file__])

if __name__ == '__main__':
    main()
