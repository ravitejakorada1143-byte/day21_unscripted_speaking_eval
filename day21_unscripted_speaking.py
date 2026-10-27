
import os
import time
from datetime import datetime

import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from groq import Groq


# Load API key from .env
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please check your .env file."
    )

client = Groq(api_key=API_KEY)

# Audio configuration
SAMPLE_RATE = 16000
DURATION = 120

# Speaking topics
TOPICS = [
    "Describe your favorite place.",
    "Talk about the importance of teamwork.",
    "Describe a memorable day in your life."
]


def choose_topic():
    """Display topics and return the user's selection."""

    print("\n=== Unscripted Speaking Evaluation ===")
    print("\nChoose a topic:")

    for number, topic in enumerate(TOPICS, start=1):
        print(f"{number}. {topic}")

    choice = input("\nEnter topic number (1-3): ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(TOPICS):
        print("Invalid choice. Please run the program again.")
        return None

    return TOPICS[int(choice) - 1]


def record_audio():
    """Record microphone audio and save it as a WAV file."""

    filename = "day21_speaking.wav"

    try:
        input("\nPress Enter when you are ready to speak...")

        print("\nRecording starts in 3 seconds...")
        time.sleep(3)

        print(f"Recording for up to {DURATION} seconds.")
        print("Speak naturally about your selected topic.")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        write(filename, SAMPLE_RATE, audio)

        print(f"\nRecording saved: {filename}")
        return filename

    except Exception as error:
        print(f"Recording failed: {error}")
        return None


def transcribe_audio(filename):
    """Transcribe the recorded audio using Groq Whisper."""

    try:
        print("\nTranscribing your speech with Groq...")

        with open(filename, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(filename), audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        if isinstance(result, str):
            transcript = result.strip()
        else:
            transcript = result.text.strip()

        print("\n=== Transcript ===")
        print(transcript)

        return transcript

    except Exception as error:
        print(f"Transcription failed: {error}")
        return None


def save_text_report(topic, filename, transcript):
    """Save the evaluation details and transcript in a text file."""

    output_file = "day21_evaluation.txt"
    word_count = len(transcript.split())

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("DAY 21 - UNSCRIPTED SPEAKING EVALUATION\n")
            file.write("=" * 45 + "\n\n")

            file.write(f"Topic: {topic}\n")
            file.write("Provider: Groq\n")
            file.write("Model: whisper-large-v3-turbo\n")
            file.write(f"Audio file: {filename}\n")
            file.write(f"Word count: {word_count}\n")
            file.write("Transcription status: Successful\n")
            file.write(
                f"Timestamp: {datetime.now().isoformat()}\n\n"
            )

            file.write("TRANSCRIPT\n")
            file.write("-" * 45 + "\n")
            file.write(transcript + "\n\n")

            file.write("FLUENCY EVALUATION\n")
            file.write("-" * 45 + "\n")
            file.write(
                "Native Azure fluency scores: Not available.\n"
            )
            file.write(
                "This implementation provides speech transcription "
                "and word count only.\n"
            )
            file.write(
                "Prosody, rhythm, and pronunciation are not "
                "directly scored by this implementation.\n"
            )

        print(f"\nText report saved: {output_file}")
        print(f"Transcript word count: {word_count}")

    except OSError as error:
        print(f"Could not save text report: {error}")


def main():
    """Run the unscripted speaking pipeline."""

    topic = choose_topic()

    if topic is None:
        return

    print(f"\nSelected topic: {topic}")

    filename = record_audio()

    if filename is None:
        return

    transcript = transcribe_audio(filename)

    if transcript:
        save_text_report(topic, filename, transcript)
        print("\nDay 21 pipeline completed successfully.")
    else:
        print("Pipeline stopped because transcription failed.")


if __name__ == "__main__":
    main()
