from speechbrain.pretrained import VAD as SpeechBrainVAD
from collections import deque
from threading import Thread
import numpy as np
import datetime
import pyaudio
import torch
import time
import wave


# Configuration
WINDOW_SIZE_MS = 800
FRAME_SIZE_MS = 10
WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_MS / FRAME_SIZE_MS)
SAMPLE_RATE = 16000
SPEECH_THRESHOLD = 0.7
NOT_SPEECH_THRESHOLD = 0.4

class StreamVAD:
    def __init__(self, wav_location="recordings/", debug=False):
        self.vad = SpeechBrainVAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmpdir")
        self.debug = debug
        self.wav_location = wav_location
        self.audio_queue = deque()
        self._STOP = False

    def log(self, message):
        if self.debug:
            print(message)

    def save_recording(self, recorded_frames, timestamp):
        filename = self. wav_location + f"speech_{timestamp}.wav"
        audio_data = b''.join(recorded_frames)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        self.log(f"Audio file written: {filename}")

    def process_stream(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=320)
        recording = False
        window_buffer = deque(maxlen=WINDOW_SIZE_FRAMES)
        recorded_frames = []
        
        while True:
            if self._STOP:
                break

            frame = stream.read(320, exception_on_overflow=False)
            window_buffer.append(frame)

            if len(window_buffer) == window_buffer.maxlen:
                window_data = b''.join(window_buffer)
                audio_tensor = torch.from_numpy(np.frombuffer(window_data, dtype=np.int16).copy()).float()
                audio_tensor = audio_tensor.unsqueeze(0)

                # Get frame-level posteriors
                prob_chunks = self.vad.get_speech_prob_chunk(audio_tensor)

                # Basic threshold to determine if speech exists.
                is_speech = torch.mean(prob_chunks) > SPEECH_THRESHOLD
                is_not_speech = torch.mean(prob_chunks) < NOT_SPEECH_THRESHOLD

                if is_speech and not recording:
                    recording = True
                    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    self.log(f"Start recording at {start_time}")
                    recorded_frames.extend(window_buffer)
                elif recording:
                    recorded_frames.append(frame)

                if recording and is_not_speech:
                    self.log("End of speech detected")
                    self.audio_queue.append((recorded_frames, start_time))
                    recorded_frames = []
                    recording = False

        # After possible breaking conditions        
        self.log("Stopping stream")
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    def start_stream(self):
        # Add support for multiple streams?
        stream_thread = Thread(target=self.process_stream)
        stream_thread.start()
    
    def stop_stream(self):
        self._STOP = True

    # Save the recordings to wave files
    def record_speech(self):
        self.start_stream()

        for recorded_frames, start_time in self:
            self.save_recording(recorded_frames, start_time)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while not self.audio_queue:
            time.sleep(0.1)
        return self.audio_queue.popleft()

if __name__ == "__main__":
    vad_processor = StreamVAD(debug=True)
    vad_processor.start_stream()
    time.sleep(10)
    vad_processor.stop_stream()