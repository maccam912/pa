import pyaudio
import wave
import whisper
import threading
import queue
import numpy as np
from textual.app import App
from textual.containers import Container
from textual.widgets import Header, Footer, Button, Static
from textual.reactive import reactive

class TranscriptionWidget(Static):
    text = reactive("")
    
    def append_text(self, new_text):
        if new_text.strip():  # Only append if there's actual text
            self.text = (self.text + " " + new_text).strip()

    def render(self):
        return self.text

class TranscriberApp(App):
    CSS = """
    Screen {
        align: center middle;
    }

    #transcription {
        width: 90%;
        height: 70%;
        border: solid green;
        margin: 2 4;
        padding: 1 2;
        background: $surface;
        overflow-y: scroll;
    }

    #controls {
        width: 90%;
        height: 10%;
        margin: 1 2;
        layout: horizontal;
        align: center middle;
    }

    Button {
        margin: 1 2;
    }
    """

    TITLE = "Real-time Audio Transcriber"

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.audio_thread = None
        self.model = None
        self.stream = None
        self.p = None

    def compose(self):
        yield Header()
        yield Container(
            TranscriptionWidget(id="transcription"),
            Container(
                Button("Start Recording", id="start", variant="success"),
                Button("Stop Recording", id="stop", variant="error"),
                id="controls"
            )
        )
        yield Footer()

    def on_mount(self):
        self.load_model()

    def load_model(self):
        self.model = whisper.load_model("base")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start" and not self.is_recording:
            self.start_recording()
        elif event.button.id == "stop" and self.is_recording:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        self.audio_thread = threading.Thread(target=self.audio_capture)
        self.transcription_thread = threading.Thread(target=self.process_audio)
        
        self.audio_thread.start()
        self.transcription_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def audio_capture(self):
        while self.is_recording:
            try:
                data = np.frombuffer(self.stream.read(1024), dtype=np.float32)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Error capturing audio: {e}")
                break

    def process_audio(self):
        audio_data = []
        while self.is_recording or not self.audio_queue.empty():
            try:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_data.append(chunk)

                    if len(audio_data) > 50:  # Process ~2 seconds of audio
                        audio_segment = np.concatenate(audio_data)
                        result = self.model.transcribe(audio_segment)
                        self.update_transcription(result["text"])
                        audio_data = []
            except Exception as e:
                print(f"Error processing audio: {e}")
                break

    def update_transcription(self, text):
        widget = self.query_one("#transcription")
        widget.append_text(text)

if __name__ == "__main__":
    app = TranscriberApp()
    app.run()
