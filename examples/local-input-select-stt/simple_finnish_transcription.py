import asyncio
import io
import os
import platform
import sys
import time
import wave
from typing import List, Optional

import numpy as np
import pyaudio
from dotenv import load_dotenv
from loguru import logger

# Load environment variables and configure logging
load_dotenv(override=True)
logger.remove()
logger.add(sys.stderr, level="INFO")

# Hardcoded parameters to keep it simple
SOURCE_LANGUAGE = "fi"  # Finnish
TARGET_LANGUAGE = "en"  # English
TTS_VOICE = "nova"      # OpenAI voice

# OpenAI API key 
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Import OpenAI API
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

class AudioRecorder:
    """A class to record audio with voice activity detection"""
    
    def __init__(self, device_index=None, sample_rate=16000, vad_threshold=500):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.vad_threshold = vad_threshold  # Simple threshold for voice detection
        self.silence_time = 1.0  # Seconds of silence to consider speech ended
        
        self.pyaudio = pyaudio.PyAudio()
        
    def _is_speaking(self, audio_data):
        """Simple voice activity detection based on amplitude"""
        data = np.frombuffer(audio_data, dtype=np.int16)
        amplitude = np.abs(data).mean()
        return amplitude > self.vad_threshold
    
    def record_with_vad(self, max_duration=30):
        """Record audio with voice activity detection"""
        logger.info(f"Opening audio stream with device {self.device_index}")
        stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        
        print("\nListening... (speak now)")
        
        frames = []
        silent_chunks = 0
        max_silent_chunks = int(self.silence_time * self.sample_rate / self.chunk_size)
        recording = False
        max_chunks = int(max_duration * self.sample_rate / self.chunk_size)
        
        # Wait for speech to start
        while True:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            if self._is_speaking(data):
                recording = True
                frames.append(data)
                print("\nSpeech detected - Recording...", end="", flush=True)
                break
            time.sleep(0.01)  # Small delay to reduce CPU usage
        
        # Record until silence is detected
        for i in range(max_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            
            if self._is_speaking(data):
                silent_chunks = 0
                if i % 10 == 0:  # Update every ~10 chunks
                    print(".", end="", flush=True)
            else:
                silent_chunks += 1
                
            if silent_chunks >= max_silent_chunks and recording:
                print("\nSilence detected - Processing speech...")
                break
                
        stream.stop_stream()
        stream.close()
        
        return frames
    
    def save_audio(self, frames, filename="temp_audio.wav"):
        """Save recorded audio to WAV file"""
        logger.info(f"Saving audio to {filename}")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        return filename
    
    def create_audio_buffer(self, frames):
        """Create in-memory audio buffer"""
        logger.info("Creating in-memory audio buffer")
        
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        buf.seek(0)
        return buf
    
    def close(self):
        """Clean up resources"""
        self.pyaudio.terminate()


def transcribe_audio(audio_buffer, language=SOURCE_LANGUAGE):
    """Transcribe audio using OpenAI Whisper API"""
    logger.info(f"Transcribing audio in language: {language}")
    
    try:
        # Pass the buffer as a tuple: (filename, file_object)
        # This helps the API identify the file format via the extension.
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_buffer),
            language=language
        )
        
        return transcription.text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


def translate_text(text, source_lang=SOURCE_LANGUAGE, target_lang=TARGET_LANGUAGE):
    """Translate text using OpenAI API"""
    if not text:
        logger.warning("Empty text, skipping translation")
        return None
    
    logger.info(f"Translating from {source_lang} to {target_lang}: '{text}'")
    
    try:
        prompt = f"""Translate the following text from {source_lang} to {target_lang}. 
        Do not explain or add anything, just return the translated text:

        {text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None


def text_to_speech(text, voice=TTS_VOICE):
    """Convert text to speech using OpenAI TTS API"""
    if not text:
        logger.warning("Empty text, skipping TTS")
        return None
    
    logger.info(f"Converting to speech with voice {voice}: '{text}'")
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        return response.content
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


def play_audio_direct(audio_data):
    """Play audio data directly through PyAudio without saving to file"""
    logger.info("Playing audio directly through PyAudio")
    print("\nPlaying translated speech...")
    
    try:
        # We need to convert MP3 to WAV for direct playback
        from pydub import AudioSegment

        # Load MP3 data
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        
        # Convert to raw PCM audio data
        pcm_data = audio_segment.raw_data
        
        # Get audio parameters
        sample_width = audio_segment.sample_width
        channels = audio_segment.channels
        frame_rate = audio_segment.frame_rate
        
        # Create PyAudio instance
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=frame_rate,
            output=True
        )
        
        # Play audio
        chunk_size = 1024
        offset = 0
        while offset < len(pcm_data):
            end = min(offset + chunk_size, len(pcm_data))
            stream.write(pcm_data[offset:end])
            offset = end
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
        
    except ImportError:
        logger.warning("pydub not installed, saving to file and playing")
        # Fallback to file-based playback but don't open external player
        temp_file = "temp_output.mp3"
        with open(temp_file, "wb") as f:
            f.write(audio_data)
            
        print(f"Audio saved to {temp_file} (pydub not installed for direct playback)")
        return False


def select_audio_device():
    """Select audio input device"""
    p = pyaudio.PyAudio()
    
    # List available audio devices
    print("\nAvailable audio input devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"  Input {i}: {dev.get('name')}")
    
    # Get default device
    try:
        input_device = p.get_default_input_device_info()['index']
    except IOError:
        input_device = 0
        print("No default input device found, using 0")
    
    # Let user select device
    input_str = input(f"\nSelect input device (default {input_device}): ")
    input_device = int(input_str) if input_str.strip() else input_device
    
    p.terminate()
    return input_device


def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print(" SIMPLE AUDIO TRANSLATOR (DIRECT VAD)")
    print(f" Translates speech from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}")
    print("="*80 + "\n")


def main():
    """Main function"""
    print_header()
    
    # Select audio device
    input_device = select_audio_device()
    
    # Create audio recorder
    recorder = AudioRecorder(device_index=input_device)
    
    try:
        while True:
            # Wait for user to press Enter to start
            input("\nPress Enter to start recording (or Ctrl+C to exit)...")
            
            # Record audio with VAD
            audio_frames = recorder.record_with_vad()
            
            # Create audio buffer
            audio_buffer = recorder.create_audio_buffer(audio_frames)
            
            # Transcribe audio using the buffer
            transcription = transcribe_audio(audio_buffer)
            
            if transcription:
                print("\n" + "="*60)
                print(f"TUNNISTETTU PUHE ({SOURCE_LANGUAGE}):")
                print(f"{transcription}")
                print("="*60)
                
                # Translate text
                translation = translate_text(transcription)
                
                if translation:
                    print("\n" + "="*60)
                    print(f"KÄÄNNETTY TEKSTI ({TARGET_LANGUAGE}):")
                    print(f"{translation}")
                    print("="*60)
                    
                    # Convert to speech
                    speech_data = text_to_speech(translation)
                    
                    if speech_data:
                        # Play audio directly
                        play_audio_direct(speech_data)
    
    except KeyboardInterrupt:
        print("\nExiting program...")
    
    finally:
        # Clean up
        recorder.close()


if __name__ == "__main__":
    main()