import asyncio
import os
import signal
import sys
from typing import Dict, Optional

import pyaudio
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Required imports from pipecat
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.openai_realtime_beta import OpenAIRealtimeBetaLLMService
from pipecat.services.openai_realtime_beta.events import (SessionProperties,
                                                          TurnDetection)
from pipecat.transports.local.audio import (LocalAudioTransport,
                                            LocalAudioTransportParams)


def select_audio_device():
    """Simple function to select audio input device"""
    p = pyaudio.PyAudio()
    
    # List available audio devices
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"  Input {i}: {dev.get('name')}")
        if dev.get('maxOutputChannels') > 0:
            print(f"  Output {i}: {dev.get('name')}")
    
    # Get default devices
    try:
        input_device = p.get_default_input_device_info()['index']
    except IOError:
        input_device = 0
        print("No default input device found, using 0")
    
    try:
        output_device = p.get_default_output_device_info()['index']
    except IOError:
        output_device = 0
        print("No default output device found, using 0")
    
    # Let user select devices
    input_str = input(f"\nSelect input device (default {input_device}): ")
    input_device = int(input_str) if input_str.strip() else input_device
    
    output_str = input(f"Select output device (default {output_device}): ")
    output_device = int(output_str) if output_str.strip() else output_device
    
    p.terminate()
    return input_device, output_device


def print_header():
    """Print application header"""
    print("\n" + "="*80)
    print(" REAALIAIKAINEN ÄÄNI CHATBOT")
    print(" Käyttää OpenAI:n Realtime API:a")
    print("="*80 + "\n")


async def main():
    """Main function to run the chatbot"""
    print_header()
    
    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Select audio devices
    input_device, output_device = select_audio_device()
    
    # Configure audio transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device,
            output_device_index=output_device,
            audio_in_channels=1,  # Mono input
            audio_in_sample_rate=16000,  # 16kHz for speech recognition
        )
    )
    
    # Configure OpenAI Realtime LLM Service
    llm = OpenAIRealtimeBetaLLMService(
        api_key=api_key,
        model="gpt-4o-realtime-preview-2024-12-17",  # Using the realtime model
        session_properties=SessionProperties(
            modalities=["audio", "text"],  # Enable both audio and text
            voice="nova",  # Choose a voice
            instructions=(
                "Olet avulias tekoälyassistentti. Vastaa käyttäjän kysymyksiin "
                "selkeästi ja ytimekkäästi. Puhu suomeksi, ellei käyttäjä pyydä "
                "erikseen muuta kieltä. Ole ystävällinen ja avulias."
            ),
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=800
            ),
            temperature=0.7  # Control the randomness of the responses
        )
    )
    
    # Build the pipeline
    pipeline = Pipeline([
        transport.input(),  # Audio input from microphone
        llm,                # Process with OpenAI Realtime LLM
        transport.output()  # Audio output to speakers
    ])
    
    # Create the pipeline task
    task = PipelineTask(pipeline)
    
    # Create the pipeline runner
    # Handle platform-specific signal handling
    import platform
    runner = PipelineRunner(handle_sigint=False if platform.system() == "Windows" else True)
    
    # Start the pipeline
    logger.info("Starting realtime voice chatbot")
    print("\nChatbot is ready. Start speaking to interact.")
    print("The chatbot will respond in real-time.")
    print("Press Ctrl+C to exit\n")
    
    try:
        await asyncio.gather(runner.run(task))
    except KeyboardInterrupt:
        print("\nStopping chatbot...")
    finally:
        # Explicitly stop the runner to clean up resources
        await runner.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")