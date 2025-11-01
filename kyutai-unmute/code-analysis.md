The `main_websocket.py` file implements a **FastAPI-based WebSocket server** that acts as the backend for a real-time audio interaction system, integrating voice-to-text (STT), text-to-speech (TTS), and large language models (LLM) — all orchestrated through a custom WebSocket-based protocol inspired by OpenAI’s Realtime API. Here's a detailed breakdown of **how the server works**, including its architecture, communication flow, and state management.

---

### 🧠 High-Level Overview

This server serves real-time voice conversations between a user and an AI model:

- Accepts live **Opus-encoded audio data over WebSocket**
- Converts it into text (**STT**: speech-to-text)
- Sends to an LLM for generating responses
- Converts the response into synthetic speech (**TTS**: text-to-speech)
- Streams back Opus-encoded responses in real time

It also supports additional endpoints like `/v1/voices`, `/v1/voice-donation`, etc., to support voice cloning or donations.

---

## 🔄 Main Steps in Handling a Connection

When a WebSocket connects to `/v1/realtime`, here's what happens step-by-step:

---

### ✅ Step 1: Upgrade to WebSocket
```python
@app.websocket("/v1/realtime")
async def websocket_route(websocket: WebSocket):
```
Accepts a new incoming connection, ensuring the correct subprotocol `"realtime"` is used (OpenAI compatibility).

Also limits concurrent active sessions (`SEMAPHORE`) to cap resource usage per instance.

---

### ✅ Step 2: Initialize Session Handler
```python
handler = UnmuteHandler()
await handler.start_up()
```
Initializes components such as:
- STT/TTS clients
- Quest manager (story/game tracking)
- Audio buffer, recorder (for logs), event queues

All services communicate with external servers via WebSockets or HTTP calls as needed.

Note: Service availability is pre-checked before accepting this connection.

---

### ✅ Step 3: Start Two Async Loops
Inside `_run_route`:
```python
tg.create_task(receive_loop(...), name="receive_loop()")
tg.create_task(emit_loop(...), name="emit_loop()")
```

Two independent async loops run concurrently inside a `TaskGroup()`:
- **Receive Loop**: Handles messages sent by the client
- **Emit Loop**: Processes and transmits events generated internally

They coordinate using shared `UnmuteHandler` and an event queue (`asyncio.Queue`).

---

## 💬 Communication Protocol – Based on OpenAI Realtime Events

Communication follows a lightweight JSON-based messaging pattern where both sides exchange predefined structures called “events”.

### 👤 Client → Server
Events sent **by the client** include:
| Event Type                    | Purpose |
|------------------------------|---------|
| `input_audio_buffer.append`  | Add new chunk of encoded (Opus) audio. |
| `session.update`             | Update configuration of current session. |

These are validated and dispatched accordingly.

#### Special Logic:
- Only valid Opus chunks marked with “first packet” flag after reconnection accepted
- Invalid/unrecognized events trigger error feedback

---

### 🤖 Server → Client
Events produced **by the server/UI handler** include:

| Event Type                  | Description |
|----------------------------|-------------|
| `conversation.item.created`| LLM generates output segment or tool invocation item |
| `response.audio.delta`     | Live synthesized TTS stream in base64-encoded Opus |
| `session.updated`          | Acknowledgement after session config change |
| `error`                     | Any kind of validation/runtime fault |
| `close`                     | End conversation cleanly |

---

## 📈 Key Transitions During Session Flow

Here’s how various stages transition during a full cycle of a call:

1. **Client Connect + Accept Handshake**
2. **Session Setup (`session.update`)**
   - Configure desired model parameters like voice, turn detection, instructions, etc.
3. **Audio Input (`input_audio_buffer.append`)**
   - Stream microphone audio continuously for transcription and processing
4. **STT → Text Transcription**
   - Server sends recognized transcription segments as items (`item.created`)
5. **Text Processed By Model**
   - AI responds, generating text replies or actions
6. **Synthesis Output Streaming**
   - Asynchronously produces small playable audio fragments (`audio.delta`) until EOF
7. **Close Stream Signal**
   - When response finishes, the internal logic emits a `CloseStream`
8. **Clean Disconnect / Reconnect Gracefully**

All these steps maintain bi-directional communication via structured payloads conforming to schema defined under `openai_realtime_api_events`.

---

## 🔧 Internals Behind Message Handling

Let’s break down each loop component more clearly.

---

### 📥 Receive Loop (`receive_loop`)

Responsible for handling inbound traffic.

Logic includes:
- Reading one message per frame (via `websocket.receive_text()`)
- Parsing JSON to validate against known Event schemas (using Pydantic's `TypeAdapter`)
- Dispatch based on message type:
  - If it's **audio data**, decode Opus → PCM, forward to handler
  - Else update session or log unknown/metrics

Additionally ensures clean cleanup upon disconnects and prevents stale messages (from prior connections) being applied post-reconnect.

---

### 📤 Emit Loop (`emit_loop`)

Handles sending outputs back toward the frontend client:

Flow:
1. Checks if any pending message exists already in the queue
   - Used mostly for immediate user alerts like errors
2. Otherwise pulls next output (either generated audio OR structured event) from `handler.emit()`
3. Encodes raw PCM samples using `OpusStreamWriter` → sends delta events containing base64 Opus-encoded audio
4. Sends textual events directly as JSON payloads with `.model_dump_json()`
5. Logs every outgoing server event to the recorder if applicable

If network disconnection detected early, will raise `WebSocketClosedError` to signal graceful shutdown across handlers/tasks.

---

## 🛠 Lifecycle Management with Cleanup

Everything wraps up when:
- One side closes the socket voluntarily or unexpectedly
- Error conditions are hit (`RuntimeError`, timeouts, validation failures)
- Hard faults result in error reporting followed by disconnect code

All resources including audio encoders, task groups, timers, service handles are cleaned up gracefully in `finally` clauses.

```python
finally:
    await handler.cleanup()
```

---

## ⚙️ Backend Dependencies Overview

### External Services Connected:
| Name                      | Role |
|--------------------------|------|
| TTS_SERVER               | Speech synthesis (Moshi, possibly Parler-TTS, etc.) |
| STT_SERVER               | Speech recognition (e.g., Whisperstream variant) |
| LLM_SERVER               | Generates conversational text (Kyutai, or OpenAI proxy) |
| VOICE_CLONING_SERVER      | Clones user voice for personalization |

Service health is periodically assessed at `/v1/health`

### Metrics Tracking:
Uses Prometheus instrumentation (via `prometheus_fastapi_instrumentator`)

Exposed metric examples:
- Sessions count
- Active sessions gauge
- Duration histogram
- Errors encountered

---

## 🗂 Middleware Layers Present

Several important pieces of middleware enhance behavior:

| Middleware                         | Use Case |
|----------------------------------|----------|
| `CORSMiddleware`                 | Allow cross-site JS access during dev |
| `LimitUploadSizeForPath`         | Cap file upload sizes for endpoints like `/v1/voices` |
| Custom Exception Handlers        | Ensure proper CORS headers even if exceptions happen |
| Optional profiling (`PROFILER`)  | Debug performance bottlenecks |

---

## ⏱ Performance Considerations

Key design choices optimize scalability without burning CPU:
- Uses threading carefully with GIL-aware `asyncio.to_thread()` only where native libraries require synchronous execution (like decoding Opus).
- Caps total simultaneous connections via semaphore (`MAX_CLIENTS = 4`).
- Leverages FastAPI's concurrency model efficiently with task grouping for parallelism.
- Logs and metrics help monitor live activity and debug issues quickly.

---

## ✨ Summary Diagram

```
Client ↔ Websocket (/v1/realtime)
          ↓
WebSocket.accept(subprotocol="realtime")
          ↓
Create UnmuteHandler() with STT/TTS/LLM clients
          ↓
Spawn two looping tasks:
    ← ReceiveLoop: Read client input_audio_buffer.append & session.update
    → EmitLoop: Write server speech deltas (audio/text) and responses
          ↓
Internal engine orchestrates voice input/output chain with model inference
          ↓
Metrics collected, recordings stored optionally, safe disconnection handled
```

---

# unmute_handler.py

Looking at `unmute_handler.py`, this is the core logic handler for managing real-time audio conversations. It orchestrates the flow between speech-to-text (STT), large language model (LLM), and text-to-speech (TTS) components. Let me break down how it works:

## **Core Architecture**

The `UnmuteHandler` class extends `AsyncStreamHandler` and serves as the central coordinator for a single conversation session. It manages:

1. **Audio processing pipeline** (STT → LLM → TTS)
2. **Conversation state management**
3. **Service lifecycle coordination**
4. **Real-time audio streaming**

## **Key Components & Workflow**

### **1. Initialization & State Management**
```python
def __init__(self) -> None:
    super().__init__(input_sample_rate=SAMPLE_RATE, output_frame_size=480, output_sample_rate=SAMPLE_RATE)
    self.n_samples_received = 0  # Audio time tracking
    self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()  # Output buffer
    self.chatbot = Chatbot()  # Manages conversation history
    self.quest_manager = QuestManager()  # Service lifecycle management
```

### **2. Service Management via QuestManager**
The handler uses a `QuestManager` to manage service lifecycles:
- **STT Quest**: Handles speech recognition
- **TTS Quest**: Handles text-to-speech synthesis  
- **LLM Quest**: Handles language model responses

### **3. Main Audio Processing Flow**

#### **Audio Reception (`receive` method)**
```python
async def receive(self, frame: tuple[int, np.ndarray]) -> None:
```
1. **Audio Input**: Receives PCM audio frames from the WebSocket
2. **STT Processing**: Sends audio to speech-to-text service
3. **Pause Detection**: Monitors for speech pauses/endings
4. **Interruption Handling**: Detects when user interrupts bot

#### **Key Detection Logic**:
- **Voice Activity Detection (VAD)**: Uses STT pause predictions
- **Silence Timeout**: Triggers after `USER_SILENCE_TIMEOUT` seconds
- **Interruption**: User can interrupt bot speech

### **4. Conversation Flow States**

#### **State Transitions**:
1. **Waiting for User** → **User Speaking** (audio detected)
2. **User Speaking** → **Processing** (pause detected)
3. **Processing** → **Bot Speaking** (LLM response + TTS)
4. **Bot Speaking** → **Waiting for User** (response complete)

#### **State Management**:
```python
# Conversation states managed in Chatbot:
# - "waiting_for_user": Bot waiting for user input
# - "user_speaking": User is talking
# - "bot_speaking": Bot is generating/responding
```

### **5. Response Generation Pipeline**

#### **Step 1: STT Processing** (`_stt_loop`)
```python
async def _stt_loop(self, stt: SpeechToText):
    async for data in stt:
        # Transcribe audio chunks
        await self.output_queue.put(ora.ConversationItemInputAudioTranscriptionDelta(...))
        # Update chat history with transcribed text
```

#### **Step 2: LLM Generation** (`_generate_response_task`)
```python
async def _generate_response_task(self):
    # Send conversation history to LLM
    async for delta in rechunk_to_words(llm.chat_completion(messages)):
        # Stream response words to TTS
        await tts.send(delta)
```

#### **Step 3: TTS Synthesis** (`_tts_loop`)
```python
async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
    async for message in tts:
        if isinstance(message, TTSAudioMessage):
            # Send audio frames to client
            await output_queue.put((SAMPLE_RATE, audio))
        elif isinstance(message, TTSTextMessage):
            # Send text deltas to client
            await output_queue.put(ora.ResponseTextDelta(delta=message.text))
```

## **Key Features**

### **Real-time Interruption**
```python
async def interrupt_bot(self):
    # Clear audio queues
    # Stop current TTS/LLM tasks
    # Prepare for new user input
```

### **Debugging & Monitoring**
- Tracks timing metrics (TTFT, RTF)
- Maintains debug plots and dictionaries
- Records session data if enabled

### **Error Handling & Recovery**
- Graceful service shutdown
- Queue management during interruptions
- Connection error recovery

## **Communication Flow**

```
User Audio → WebSocket → receive() → STT Service → 
Transcription → Chatbot History → LLM Request → 
LLM Response → TTS Service → Audio Response → 
WebSocket → Client
```

## **Service Lifecycle**

### **Startup Sequence**:
1. Initialize STT service (`start_up_stt`)
2. Wait for STT readiness
3. Begin audio processing

### **Dynamic Service Management**:
- Services created/deleted as needed
- Error recovery with retry logic
- Clean shutdown procedures

The handler essentially acts as a **real-time conversation orchestrator** that manages the complex timing and state transitions required for natural voice interactions, while handling interruptions, timeouts, and service coordination.

# quest_manager.py

Looking at `quest_manager.py`, this is a resource management system that provides **RAII (Resource Acquisition Is Initialization)** patterns in Python for handling asynchronous services with proper cleanup. Let me break down how it works:

## **Core Concept**

The `QuestManager` implements a **structured concurrency pattern** for managing the lifecycle of asynchronous services (STT, TTS, LLM connections) with guaranteed cleanup, similar to try/finally blocks but more sophisticated.

## **Key Components**

### **Quest[T] Class**
A `Quest` represents a unit of async work with three phases:
1. **Init**: Initialize the resource (e.g., connect to STT service)
2. **Run**: Use the resource (e.g., process audio stream)
3. **Close**: Clean up the resource (e.g., disconnect)

```python
class Quest[T]:
    def __init__(
        self,
        name: str,
        init: Callable[[], Awaitable[T]],  # Resource initialization
        run: Callable[[T], Awaitable[None]],  # Resource usage
        close: Callable[[T], Awaitable[None]] | None = None,  # Cleanup
    ):
```

### **QuestManager Class**
Manages multiple quests with automatic conflict resolution and cleanup:

```python
class QuestManager:
    def __init__(self):
        self.quests: dict[str, Quest] = {}  # Named quests
        self._future: asyncio.Future | None = None  # Aggregates exceptions
```

## **How It Works**

### **1. Quest Lifecycle**

#### **Creation & Initialization**
```python
# Example: Creating an STT quest
async def _init() -> SpeechToText:
    return await find_instance("stt", SpeechToText)  # Connect to service

async def _run(stt: SpeechToText):
    await self._stt_loop(stt)  # Process audio stream

async def _close(stt: SpeechToText):
    await stt.shutdown()  # Clean disconnect

quest = Quest("stt", _init, _run, _close)
```

#### **Starting a Quest**
```python
async def __aenter__(self) -> asyncio.Future[None]:
    self.task = asyncio.create_task(self._run())  # Start the quest
    return asyncio.ensure_future(self.task)
```

### **2. Quest Management**

#### **Adding Quests**
```python
async def add(self, quest: Quest[T]) -> Quest[T]:
    name = quest.name
    # Cancel existing quest with same name
    if name in self.quests:
        await self.quests[name].remove()
    
    self.quests[name] = quest
    # Start the quest and monitor for exceptions
    future = await quest.__aenter__()
    future.add_done_callback(partial(self._one_is_done, name, self._future))
    return quest
```

#### **Automatic Replacement**
If you add a quest with a name that already exists, the old one is automatically cancelled and cleaned up.

### **3. Exception Handling**

#### **Aggregated Error Monitoring**
```python
@staticmethod
def _one_is_done(name: str, agg_future: asyncio.Future, future: asyncio.Future):
    try:
        future.result()  # Check for exceptions
    except asyncio.CancelledError:
        logger.debug("Quest %s was cancelled.", name)
    except Exception as exc:
        logger.debug("Quest %s failed with %r.", name, exc)
        if not agg_future.done():
            agg_future.set_exception(exc)  # Bubble up to main task
```

#### **Usage Pattern**
```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(handler.quest_manager.wait())  # Catches quest exceptions
```

### **4. Cleanup Guarantees**

#### **Context Manager Pattern**
```python
async def __aexit__(self, exc_type, exc_val, exc_tb):
    # Clean shutdown of all quests
    for name, quest in self.quests.items():
        try:
            await quest.remove()  # Calls close() and cancels task
        except Exception:
            logger.exception(f"Error shutting down quest {name}")
    
    self.quests.clear()
    if not self._future.done():
        self._future.set_result(None)  # Signal completion
```

## **Real-World Usage Example**

In `unmute_handler.py`, quests manage services:

```python
# STT Quest
async def start_up_stt(self):
    async def _init() -> SpeechToText:
        return await find_instance("stt", SpeechToText)
    
    async def _run(stt: SpeechToText):
        await self._stt_loop(stt)  # Process audio
    
    async def _close(stt: SpeechToText):
        await stt.shutdown()
    
    # Add quest - automatically handles replacement/cleanup
    quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
```

## **Key Benefits**

1. **Automatic Cleanup**: Resources are always cleaned up, even on errors
2. **Conflict Resolution**: Adding a quest with existing name cancels the old one
3. **Exception Propagation**: Errors in any quest bubble up to main task
4. **Structured Lifecycle**: Clear init/run/close phases prevent resource leaks
5. **Cancellation Safety**: Proper task cancellation without leaving resources hanging

This system provides robust service lifecycle management essential for real-time audio processing where services need to be dynamically created, replaced, and cleaned up without leaks or zombie processes.

# STT protocol

Based on the code structure and patterns in your files, the STT (Speech-to-Text) service communication follows a WebSocket-based protocol with custom message types. Let me trace through the detailed protocol:

## **STT Service Communication Protocol**

### **1. Connection Establishment**
The STT service is discovered and connected via:
```python
# In `quest_manager.py` and `unmute_handler.py`
async def _init() -> SpeechToText:
    return await find_instance("stt", SpeechToText)
```

This uses service discovery to find and connect to the STT WebSocket server.

### **2. Message Flow**

#### **Client → STT Service Messages**
The main client-to-STT communication happens through audio streaming:
```python
# In `unmute_handler.py` receive() method
await stt.send_audio(array)
```

This sends PCM audio chunks to the STT service. The audio is likely wrapped in a protocol like:
- Raw PCM frames
- Timestamped audio packets
- Session/context identifiers

#### **STT Service → Client Messages**
The STT service sends responses through async iteration:
```python
# In `unmute_handler.py` _stt_loop method
async for data in stt:
    if isinstance(data, STTMarkerMessage):
        # Ignore marker messages
        continue
    
    # Send transcription to client
    await self.output_queue.put(
        ora.ConversationItemInputAudioTranscriptionDelta(
            delta=data.text,
            start_time=data.start_time,
        )
    )
```

### **3. Message Types**

Based on the code, the STT service likely uses these message types:

#### **Input Messages (Client → STT)**:
- **Audio Data**: PCM audio frames
- **Session Control**: Start/stop transcription signals
- **Configuration**: Language, model selection, etc.

#### **Output Messages (STT → Client)**:
```python
# From `stt/speech_to_text.py` (inferred structure):
class STTMarkerMessage:
    type: str = "marker"
    # Internal synchronization markers

class STTTranscriptionMessage:
    type: str = "transcription"
    text: str
    start_time: float
    end_time: float
    confidence: float
```

### **4. Protocol Flow**

#### **Connection Phase**:
1. **Service Discovery**: `find_instance("stt", SpeechToText)`
2. **WebSocket Connection**: Establish connection to STT server
3. **Session Initialization**: Send configuration (sample rate, language, etc.)
4. **Ready Signal**: STT service acknowledges readiness

#### **Streaming Phase**:
```python
# Continuous loop in _stt_loop
async for data in stt:
    # Real-time transcription results
    # Partial results streamed as they become available
    # Final results marked with confidence scores
```

#### **Audio Streaming**:
```python
# From unmute_handler.py
await stt.send_audio(array)  # Send PCM audio chunks
```

The protocol likely batches audio into frames and sends them continuously.

### **5. Control Mechanisms**

#### **Pause Detection**:
```python
# In unmute_handler.py
if stt.pause_prediction.value > 0.6:
    # Voice Activity Detection triggers pause detection
    return True
```

The STT service likely sends VAD (Voice Activity Detection) scores along with transcriptions.

#### **Flushing Mechanism**:
```python
# When pause detected, send silence to flush buffers
num_frames = int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
for _ in range(num_frames):
    await stt.send_audio(zero)
```

### **6. Error Handling Protocol**

#### **Connection Errors**:
```python
# In _stt_loop
except websockets.ConnectionClosed:
    logger.info("STT connection closed while receiving messages.")
```

#### **Service Unavailability**:
```python
# In main_websocket.py via QuestManager
except MissingServiceAtCapacity:
    # Handle service overload
except MissingServiceTimeout:
    # Handle connection timeouts
```

### **7. Session Management**

#### **Start Session**:
- Connect to WebSocket
- Send initial configuration
- Begin audio streaming

#### **End Session**:
```python
# In Quest cleanup
async def _close(stt: SpeechToText):
    await stt.shutdown()  # Graceful shutdown
```

### **8. Timing and Synchronization**

#### **Delay Compensation**:
```python
# Account for processing delays
self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
```

#### **Real-time Factor (RTF) Monitoring**:
```python
rtf = stt.delay_sec / elapsed  # Processing speed vs real-time
```

### **9. Inferred Protocol Message Structure**

While the exact STT protocol isn't shown, based on the patterns, it likely uses JSON messages:

```json
// Audio packet
{
  "type": "audio",
  "data": "<base64_encoded_pcm>",
  "timestamp": 1234567890.123,
  "sample_rate": 24000
}

// Transcription result (partial)
{
  "type": "transcription",
  "text": "hello world",
  "start_time": 1.234,
  "end_time": 2.345,
  "confidence": 0.95,
  "is_final": false
}

// Transcription result (final)
{
  "type": "transcription",
  "text": "hello world",
  "start_time": 1.234,
  "end_time": 2.345,
  "confidence": 0.98,
  "is_final": true
}

// VAD update
{
  "type": "vad",
  "probability": 0.85,
  "speaking": true
}

// Session control
{
  "type": "control",
  "action": "flush",
  "duration": 0.1
}
```

### **10. Key Protocol Features**

1. **Real-time Streaming**: Continuous audio input with immediate partial results
2. **VAD Integration**: Voice activity detection for pause/speech boundaries
3. **Confidence Scoring**: Quality indicators for transcription accuracy
4. **Session Boundaries**: Clear start/end of transcription sessions
5. **Error Recovery**: Automatic reconnection and error handling
6. **Timing Synchronization**: Timestamps for accurate alignment with other services

The protocol is designed for low-latency, real-time transcription with robust error handling and synchronization with the overall conversation flow managed by the UnmuteHandler.

# TTS protocol

Based on the code in `unmute_handler.py` and the patterns shown, the TTS (Text-to-Speech) service communication follows a more complex bidirectional protocol than STT. Let me trace through the detailed protocol:

## **TTS Service Communication Protocol**

### **1. Connection Establishment**
```python
# In `unmute_handler.py` start_up_tts method
async def _init() -> TextToSpeech:
    factory = partial(
        TextToSpeech,
        recorder=self.recorder,
        get_time=self.audio_received_sec,
        voice=self.tts_voice,
    )
    tts = await find_instance("tts", factory)
```

This creates a `TextToSpeech` instance that handles the WebSocket communication.

### **2. Message Flow**

#### **Client → TTS Service Messages**
The main client-to-TTS communication sends text for synthesis:

```python
# In `_generate_response_task` method
await tts.send(delta)  # Send text chunks to TTS

# At end of LLM response
await tts.send(TTSClientEosMessage())  # End of stream signal
```

#### **TTS Service → Client Messages**
The TTS service sends responses through async iteration:
```python
# In `_tts_loop` method
async for message in tts:
    if isinstance(message, TTSAudioMessage):
        # Send audio frames to client
        await output_queue.put((SAMPLE_RATE, audio))
    elif isinstance(message, TTSTextMessage):
        # Send text deltas (for display)
        await output_queue.put(ora.ResponseTextDelta(delta=message.text))
```

### **3. Message Types**

Based on the code, the TTS service uses specific message types:

#### **Input Messages (Client → TTS)**:
```python
# From `tts/text_to_speech.py` (inferred):
class TTSClientTextMessage:
    type: str = "text"
    text: str  # Text to synthesize

class TTSClientEosMessage:
    type: str = "eos"  # End of stream
```

#### **Output Messages (TTS → Client)**:
```python
class TTSAudioMessage:
    type: str = "audio"
    pcm: list[float]  # PCM audio samples

class TTSTextMessage:
    type: str = "text"
    text: str  # Text being synthesized (for display)
```

### **4. Protocol Flow**

#### **Connection Phase**:
1. **Service Discovery**: `find_instance("tts", factory)`
2. **WebSocket Connection**: Establish connection to TTS server
3. **Session Initialization**: Send voice configuration, sample rate, etc.
4. **Ready Signal**: TTS service acknowledges readiness

#### **Synthesis Phase**:
```python
# In `_tts_loop` method
async for message in tts:
    # Stream audio and text responses back to client
    # Audio messages are PCM samples
    # Text messages are for displaying current synthesis
```

#### **Text Streaming**:
```python
# From _generate_response_task
async for delta in rechunk_to_words(llm.chat_completion(messages)):
    # Stream response words to TTS as they arrive
    await tts.send(delta)
```

### **5. Control Mechanisms**

#### **End of Stream Signaling**:
```python
# Signal end of LLM response to TTS
await tts.send(TTSClientEosMessage())
```

#### **Real-time Timing**:
```python
# Timing measurements for throughput monitoring
time_since_start = self.audio_received_sec() - audio_started
time_received = tts.received_samples / self.input_sample_rate
self.debug_dict["tts_throughput"] = {
    "time_received": round(time_received, 2),
    "time_since_start": round(time_since_start, 2),
    "ratio": round(time_received_yielded / (time_since_start + 0.01), 2),
}
```

### **6. Error Handling Protocol**

#### **Connection Errors**:
```python
# In _tts_loop
except websockets.ConnectionClosedError as e:
    logger.error(f"TTS connection closed with an error: {e}")
```

#### **Retry Logic**:
```python
# In start_up_tts with retry mechanism
for trial in range(trials):
    try:
        tts = await find_instance("tts", factory)
    except Exception:
        if trial == trials - 1:
            raise
        await asyncio.sleep(sleep_time)
        # Send warning to client about latency
        error = make_ora_error(
            type="warning",
            message="Looking for the resources, expect some latency.",
        )
        await self.output_queue.put(error)
```

### **7. Session Management**

#### **Start Session**:
```python
# In start_up_tts method
quest = await self.quest_manager.add(Quest("tts", _init, _run, _close))
```

#### **End Session**:
```python
# In Quest cleanup
async def _close(tts: TextToSpeech):
    await tts.shutdown()  # Graceful shutdown
```

### **8. Interruption Handling**

#### **Bot Interruption**:
```python
# In interrupt_bot method
async def interrupt_bot(self):
    # Clear TTS queue and stop current synthesis
    self.output_queue = asyncio.Queue()  # Clear our own queue
    await self.quest_manager.remove("tts")  # Stop and cleanup TTS quest
    await self.quest_manager.remove("llm")   # Stop and cleanup LLM quest
```

### **9. Timing and Synchronization**

#### **Audio Timing**:
```python
# Track when audio synthesis starts
if audio_started is None:
    audio_started = self.audio_received_sec()

# Calculate real-time factor for performance monitoring
self.debug_dict["tts_throughput"] = {
    "ratio": round(time_received_yielded / (time_since_start + 0.01), 2),
}
```

### **10. Protocol Message Structure**

Based on the patterns in the code, the TTS protocol likely uses JSON messages:

```json
// Client sends text for synthesis
{
  "type": "text",
  "text": "Hello, how can I help you today?",
  "timestamp": 1234567890.123
}

// Client signals end of input
{
  "type": "eos",
  "timestamp": 1234567895.456
}

// TTS service sends audio response
{
  "type": "audio",
  "pcm": [0.1, -0.2, 0.3, ...],  // PCM samples
  "sample_rate": 24000,
  "timestamp": 1234567890.789
}

// TTS service sends text for display (as it's being synthesized)
{
  "type": "text",
  "text": "Hello, how can I help",
  "timestamp": 1234567890.789
}

// Configuration message (on connection)
{
  "type": "config",
  "voice": "alloy",
  "sample_rate": 24000,
  "language": "en"
}

// Error message
{
  "type": "error",
  "code": "VOICE_NOT_AVAILABLE",
  "message": "Requested voice is not available"
}
```

### **11. Key Protocol Features**

1. **Bidirectional Streaming**: 
   - Client streams text chunks to TTS
   - TTS streams audio and text responses back

2. **Real-time Synthesis**: 
   - Audio generated as text arrives
   - Low-latency streaming output

3. **Voice Selection**: 
   - Voice configuration sent on connection
   - Support for multiple voice options

4. **Flow Control**: 
   - EOS signaling for clean session ending
   - Interruption handling for dynamic conversations

5. **Synchronization**: 
   - Timestamps for accurate timing
   - Sample counting for throughput monitoring

6. **Error Recovery**: 
   - Retry mechanisms for service discovery
   - Graceful degradation with user warnings

7. **Interruption Support**: 
   - Ability to cancel ongoing synthesis
   - Queue clearing for immediate response changes

The TTS protocol is designed for real-time, low-latency speech synthesis with support for dynamic interruptions and robust error handling, making it suitable for conversational AI applications.

# LLM protocol

Based on the code in `unmute_handler.py` and related files, the LLM (Large Language Model) communication follows the **OpenAI API protocol** with some custom extensions. Let me break down the detailed protocol:

## **LLM Service Communication Protocol**

### **1. Connection & Client Setup**
```python
# In `unmute_handler.py` __init__ method
self.openai_client = get_openai_client()

# In `llm_utils.py` (inferred)
def get_openai_client():
    return AsyncOpenAI(
        base_url=LLM_SERVER,  # e.g., "http://localhost:8000/v1"
        api_key=KYUTAI_LLM_API_KEY,
    )
```

The system uses the standard OpenAI Python client connecting to a vLLM-compatible server.

### **2. Message Flow**

#### **Client → LLM Service Messages**
The communication uses the OpenAI Chat Completions API format:

```python
# In `_generate_response_task` method
messages = self.chatbot.preprocessed_messages()
llm = VLLMStream(self.openai_client, temperature=0.7)

async for delta in rechunk_to_words(llm.chat_completion(messages)):
    # Stream response tokens/words to TTS
    await tts.send(delta)
```

#### **LLM Service → Client Messages**
Streaming responses using Server-Sent Events (SSE) format typical of OpenAI API:

```python
# From llm_utils.py (inferred structure)
class VLLMStream:
    async def chat_completion(self, messages):
        stream = await self.client.chat.completions.create(
            model="default",  # or configured model
            messages=messages,
            stream=True,
            temperature=self.temperature,
            # other OpenAI parameters
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### **3. Message Types & Structure**

#### **Input Messages (Client → LLM)**:
Standard OpenAI Chat API format:
```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant..."
    },
    {
        "role": "user", 
        "content": "Hello, how are you?"
    },
    {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!"
    }
]
```

#### **Parameters**:
```python
# From _generate_response_task
llm.chat_completion(
    messages=messages,
    model="default",  # Configurable
    temperature=FIRST_MESSAGE_TEMPERATURE,  # 0.7 for first, 0.3 for follow-ups
    stream=True,  # Always streaming
    # Potentially other OpenAI params like max_tokens, top_p, etc.
)
```

### **4. Protocol Flow**

#### **Connection Phase**:
1. **HTTP Connection**: REST API call to `/v1/chat/completions`
2. **Authentication**: Bearer token in Authorization header
3. **Session Setup**: Send conversation history and parameters

#### **Streaming Phase**:
```python
# In _generate_response_task
async for delta in rechunk_to_words(llm.chat_completion(messages)):
    # Process each token/word as it arrives
    await self.output_queue.put(
        ora.UnmuteResponseTextDeltaReady(delta=delta)
    )
    
    # Send to TTS for real-time synthesis
    await tts.send(delta)
```

#### **Word Chunking**:
```python
# From llm_utils.py (inferred)
async def rechunk_to_words(stream):
    buffer = ""
    async for token in stream:
        buffer += token
        # Split on word boundaries for better TTS streaming
        words = buffer.split()
        if len(words) > 1:  # At least one complete word
            for word in words[:-1]:  # Send all but last (possibly incomplete)
                yield word + " "
            buffer = words[-1]  # Keep last word for next iteration
    
    # Send remaining buffer
    if buffer:
        yield buffer
```

### **5. Control Mechanisms**

#### **Temperature Control**:
```python
# Different temperatures for conversation flow
temperature=FIRST_MESSAGE_TEMPERATURE if generating_message_i == 2 else FURTHER_MESSAGES_TEMPERATURE
# FIRST_MESSAGE_TEMPERATURE = 0.7 (more creative)
# FURTHER_MESSAGES_TEMPERATURE = 0.3 (more focused)
```

#### **Interruption Handling**:
```python
# In _generate_response_task
if len(self.chatbot.chat_history) > generating_message_i:
    break  # We've been interrupted
```

#### **Cancellation Support**:
```python
# Task cancellation propagates to LLM stream
except asyncio.CancelledError:
    mt.VLLM_INTERRUPTS.inc()
    raise
```

### **6. Error Handling Protocol**

#### **Connection Errors**:
The OpenAI client handles standard HTTP errors, timeouts, etc.

#### **Service Unavailability**:
```python
# In main_websocket.py health check
llm_up = tg.create_task(
    asyncio.to_thread(
        _check_server_status,
        _ws_to_http(LLM_SERVER) + "/v1/models",
        headers={"Authorization": f"Bearer {KYUTAI_LLM_API_KEY}"},
    )
)
```

### **7. Session Management**

#### **Conversation Context**:
```python
# Full conversation history sent with each request
messages = self.chatbot.preprocessed_messages()

# Example structure:
[
    {"role": "system", "content": "System instructions..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    # Current assistant message being generated
    {"role": "assistant", "content": ""}
]
```

#### **Message Preprocessing**:
```python
# In Chatbot class (inferred)
def preprocessed_messages(self):
    # Apply any preprocessing rules
    # Handle special markers like USER_SILENCE_MARKER
    # Apply conversation context rules
    return processed_messages
```

### **8. Timing and Metrics**

#### **Performance Monitoring**:
```python
# Time to First Token tracking
llm_stopwatch = Stopwatch()
# ... start streaming
if time_to_first_token is None:
    time_to_first_token = llm_stopwatch.time()
    self.debug_dict["timing"]["to_first_token"] = time_to_first_token
    mt.VLLM_TTFT.observe(time_to_first_token)  # Prometheus metrics
```

#### **Response Analysis**:
```python
# Track response length and generation time
mt.VLLM_REPLY_LENGTH.observe(len(response_words))
mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())
mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)  # Input length
```

### **9. Protocol Message Examples**

#### **HTTP Request**:
```http
POST /v1/chat/completions
Authorization: Bearer sk-...
Content-Type: application/json

{
  "model": "default",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like?"}
  ],
  "temperature": 0.7,
  "stream": true
}
```

#### **Streaming Response (SSE)**:
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"The"},"index":0}]}

data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":" weather"},"index":0}]}

data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":" today"},"index":0}]}

data: {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

### **10. Custom Extensions**

#### **Special Markers Handling**:
```python
# In Chatbot preprocessing
if message.get("content") == USER_SILENCE_MARKER:
    # Handle long silence scenario with special prompt instructions
```

#### **Response Wrapping**:
```python
# In _generate_response_task
await self.output_queue.put(
    ora.ResponseCreated(
        response=ora.Response(
            status="in_progress",
            voice=self.tts_voice or "missing",
            chat_history=self.chatbot.chat_history,
        )
    )
)
```

### **11. Key Protocol Features**

1. **OpenAI-Compatible**: Uses standard Chat Completions API
2. **Streaming**: Real-time token streaming with SSE
3. **Contextual**: Full conversation history maintained
4. **Configurable**: Temperature and other parameters adjusted per message
5. **Interruption-Aware**: Stops generation when interrupted
6. **Metrics-Rich**: Detailed timing and performance tracking
7. **Error-Resilient**: Standard HTTP error handling
8. **Word-Boundary Optimized**: `rechunk_to_words` for better TTS streaming

The protocol leverages the widely-adopted OpenAI API standard while adding custom streaming optimizations and conversation management features specific to real-time voice interactions.

# Mutiple users and sessions

Looking at the code, the system uses several mechanisms to manage distinct sessions for multiple users while sharing some resources appropriately. Let me break down how session isolation and resource management work:

## **Session Isolation Mechanisms**

### **1. Per-Connection Handler Instances**

Each WebSocket connection gets its own **dedicated `UnmuteHandler` instance**:

```python
# In main_websocket.py
async def websocket_route(websocket: WebSocket):
    async with SEMAPHORE:
        try:
            await websocket.accept(subprotocol="realtime")
            
            # NEW INSTANCE PER CONNECTION
            handler = UnmuteHandler()  # ← Fresh instance for each user
            async with handler:
                await handler.start_up()
                await _run_route(websocket, handler)
```

This ensures complete isolation - each user has their own:
- Conversation state (`chatbot.chat_history`)
- Audio processing queues
- Service connections (STT, TTS, LLM)
- Timing trackers and debug data

### **2. Semaphore-Based Connection Limiting**

```python
# Global semaphore limits concurrent sessions
MAX_CLIENTS = 4
SEMAPHORE = asyncio.Semaphore(MAX_CLIENTS)

# In websocket_route
async with SEMAPHORE:  # Limits total concurrent users
    # ... create handler and process session
```

This prevents resource exhaustion while ensuring each session gets dedicated resources.

### **3. Independent Service Connections**

Each handler creates **independent connections** to backend services:

```python
# In UnmuteHandler.start_up_stt()
async def _init() -> SpeechToText:
    return await find_instance("stt", SpeechToText)  # ← Unique STT connection

# In UnmuteHandler.start_up_tts()  
async def _init() -> TextToSpeech:
    factory = partial(TextToSpeech, ...)  # ← Unique TTS connection
    return await find_instance("tts", factory)
```

Services like STT/TTS are allocated independently per session via service discovery.

## **What is SHARED Between Users**

### **1. Configuration Constants**
```python
# Shared across all sessions
SAMPLE_RATE = 24000
FRAME_TIME_SEC = 0.02
MAX_CLIENTS = 4
CORS_ALLOW_ORIGINS = ["http://localhost", "http://localhost:3000"]
```

### **2. Service Discovery Mechanism**
```python
# Shared service discovery pool
@async_ttl_cached(ttl_sec=0.5)  
async def _get_health(_none: None):
    # Shared health checking but separate actual connections per session
```

### **3. Global Metrics Collection**
```python
# Shared Prometheus metrics (with proper atomic operations)
mt.SESSIONS.inc()        # Global counter
mt.ACTIVE_SESSIONS.inc() # Global gauge
mt.HEALTH_OK.observe()   # Shared histogram
```

### **4. Cached Static Data**
```python
# Shared across sessions (immutable)
@app.get("/v1/voices")
@cache
def voices():  # Voice list cached globally
    return good_voices

# Health checks also shared with caching
@partial(async_ttl_cached, ttl_sec=0.5)
async def _get_health(_none: None):
```

## **What is SPECIFIC Per User Session**

### **1. Complete Handler State**
Each `UnmuteHandler` instance maintains independent:
```python
class UnmuteHandler:
    def __init__(self) -> None:
        self.n_samples_received = 0           # Per-session timing
        self.output_queue = asyncio.Queue()   # Per-session output queue
        self.chatbot = Chatbot()              # Per-session conversation
        self.quest_manager = QuestManager()   # Per-session service manager
        self.recorder = Recorder(...)         # Per-session recording (if enabled)
        self.debug_dict = {}                  # Per-session debug data
```

### **2. Independent Service Quests**
```python
# Each session has its own QuestManager with isolated services
self.quest_manager = QuestManager()
# Creates independent STT, TTS, LLM connections
```

### **3. Separate Audio Processing**
```python
# Per-session audio state tracking
self.n_samples_received = 0
self.stt_last_message_time = 0
self.waiting_for_user_start_time = self.audio_received_sec()
```

### **4. Individual WebSocket Connection**
```python
# Each session uses its own WebSocket
async def _run_route(websocket: WebSocket, handler: UnmuteHandler):
    # websocket is unique per connection
    # handler is unique per session
```

## **Session Lifecycle Isolation**

### **Creation**
```python
# New handler instance per connection
handler = UnmuteHandler()  # Completely fresh state

# Dedicated service connections
await handler.start_up()  # Creates STT connection for THIS session only
```

### **Execution**
```python
# Independent processing loops
tg.create_task(receive_loop(websocket, handler, emit_queue))
tg.create_task(emit_loop(websocket, handler, emit_queue))
tg.create_task(handler.quest_manager.wait())  # Independent service management
```

### **Destruction**
```python
# Guaranteed cleanup per session
finally:
    await handler.cleanup()  # Closes session-specific resources
    mt.ACTIVE_SESSIONS.dec()  # Updates global metric only
```

## **Resource Sharing Strategy**

### **Shared Read-Only Resources**:
- Configuration constants
- Voice lists
- Health check utilities
- Metrics collectors
- Static file serving endpoints

### **Isolated Mutable Resources**:
- Conversation state (`chatbot.chat_history`)
- Audio processing state (`n_samples_received`, timers)
- Service connections (STT, TTS, LLM)
- Recording files and handles
- Debug information and logs

### **Controlled Resource Pooling**:
- **Service instances**: Pooled at the infrastructure level but allocated uniquely per session
- **Connections**: Each session gets dedicated connections to avoid cross-talk
- **Memory**: Session-scoped to prevent leaks
- **CPU/GIL**: Semaphore limiting prevents excessive concurrency

## **Race Condition Prevention**

### **Per-Session Locks**:
```python
# Turn transition lock is per-handler (per-session)
self.turn_transition_lock = asyncio.Lock()
```

### **Independent Queues**:
```python
# Each session has its own output queue
self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
```

### **Session-Specific State**:
All mutable state is contained within the handler instance, preventing cross-session contamination.

## **Scalability Design**

The architecture is designed for horizontal scaling:
- **Vertical scaling limit**: Semaphore limits sessions per instance
- **Horizontal scaling**: Run multiple backend instances behind load balancer
- **Resource isolation**: No session-to-session interference
- **Stateless core**: Most shared resources are read-only configuration

This design ensures that each user gets a completely independent conversation experience while efficiently sharing immutable resources and infrastructure.

# Concurrent requests for STT

Looking at `batched_asr.py`, the system manages concurrent requests and GPU batching through a sophisticated batched processing approach. Here's how it works:

## Core Architecture

### 1. **Fixed-Size Batch Processing**
The system operates on **fixed batch sizes** established at initialization:
```python
service = ASRService(batch_size=batch_size, ...)
```

All operations process exactly `batch_size` items simultaneously, regardless of how many active clients there are.

### 2. **Client Slot Management**
Each batch slot represents a "client slot" managed by `ClientState`:
```python
@dataclass
class ClientState:
    is_complete: bool = False
    active: bool = False
    offset: int = 0
    real_end: int = 0

def __post_init__(self):
    self.clients = [ClientState() for _ in range(self.batch_size)]
```

Each slot can be in different states:
- **Inactive**: No active request
- **Active**: Processing audio data
- **Complete**: Waiting for final processing after EOS marker

### 3. **Dynamic Execution Masking**
The key innovation is **execution masking** that enables efficient sparse batching:

```python
def step(self, ..., updates: list[int]) -> None:
    # Build execution mask based on active clients
    exec_mask = torch.tensor(
        [client.active for client in self.clients],
        dtype=torch.bool,
        device=self.device,
    )
    
    # Only process active slots
    self.lm_gen.set_exec_mask(exec_mask)
    self.mimi.set_exec_mask(exec_mask)
```

This means:
- **Inactive slots** consume minimal GPU resources (masked out)
- **Only active slots** perform actual computation
- **Batch operations** still run on full tensors, but masked portions do minimal work

### 4. **State Management for Concurrent Requests**
The system handles concurrent requests through:

**Request Lifecycle:**
1. **New Request**: Slot marked as `ACTIVE`
2. **Ongoing Processing**: Continuous audio chunks processed
3. **End Signal**: Client sends marker, system schedules final processing
4. **Completion**: After delay (`asr_delay_in_tokens`), slot marked as `IS_EOS`

```python
# Handle different update types
if update > 0:  # End marker received
    self.clients[batch_idx].is_complete = True
    self.clients[batch_idx].real_end = (
        self.clients[batch_idx].offset 
        + update + self.asr_delay_in_tokens + 2
    )
```

### 5. **GPU Memory and Performance Optimization**
```python
# Warmup pre-allocates GPU memory for batched operations
def __post_init__(self):
    self.lm_gen.streaming_forever(self.batch_size)
    self.mimi.streaming_forever(self.batch_size)
    
    # Pre-warm models with dummy data
    for _ in range(3):
        batch_zeros = torch.zeros((self.batch_size, 1, self.mimi.frame_size))
        audio_tokens = self.mimi.encode(batch_zeros)
        frame = self.lm_gen.step(audio_tokens)
```

This ensures:
- **Consistent GPU memory usage** (fixed batch size)
- **Eliminates dynamic allocation** during runtime
- **Optimal tensor operations** on GPU

## Request Flow Example

1. **Batch arrives** with mixed active/inactive slots
2. **Execution mask** filters to only active slots
3. **Mimi model** encodes audio for all slots simultaneously:
   ```python
   batch_pcm = batch_pcm.view(self.batch_size, frame_size).unsqueeze(1)
   audio_tokens = self.mimi.encode(batch_pcm)  # Full batch operation
   ```
4. **LM model** generates text with execution masking:
   ```python
   text_tokens, extra_heads = self.lm_gen.step_with_extra_heads(audio_tokens)
   ```
5. **State updates** for each slot based on progress
6. **Results returned** for all slots simultaneously

## Key Benefits

- **Maximized GPU Utilization**: Batched operations even with sparse activity
- **Predictable Performance**: Fixed memory and computation patterns
- **Scalable Concurrency**: Batch size determines maximum concurrent requests
- **Efficient Resource Use**: Inactive slots consume minimal GPU resources

The system essentially trades some theoretical efficiency (processing empty slots) for practical benefits like predictable performance and simplified GPU memory management.

# Concurrent requests for TTS

Looking at `tts.rs`, the GPU batching and concurrent request handling is quite different from the ASR system. Here's how it works:

## Single Request Processing Model

### 1. **Mutex-Based Concurrency Control**
The TTS system uses a simple mutex to ensure **only one inference can run at a time**:

```rust
pub struct Model {
    // ... other fields
    pub(crate) mutex: tokio::sync::Mutex<()>,  // Dummy way to ensure only single inference
}

pub async fn handle_socket(&self, socket: ws::WebSocket, query: crate::TtsStreamingQuery) -> Result<()> {
    let _guard = self.mutex.lock().await;  // Only one request processed at a time
    // ... processing logic
}
```

This means:
- **No batched parallel processing** like in ASR
- **Sequential request handling** - each request blocks others
- **Single GPU context** used per request

### 2. **Per-Request GPU State Management**
Each TTS request creates its own independent state:

```rust
let mut state = moshi::tts_streaming::State::new(
    self.lm.clone(),                    // Shared model weights
    Some(moshi::transformer::CaSrc::Tokens(ca_src)),  // Per-request voice conditioning
    max_seq_len,
    audio_lp, text_lp,
    query.cfg_alpha,
    config.clone(),
);
```

The model weights are shared, but each request has:
- **Independent generation state**
- **Independent voice conditioning** (ca_src)
- **Independent logits processors** (sampling parameters)

### 3. **Streaming Pipeline Architecture**
For WebSocket streaming, the system uses a **three-thread pipeline**:

```rust
let recv_loop = tokio::task::spawn(async move { /* Socket input */ });
let process_loop = tokio::task::spawn_blocking(move { /* TTS generation */ });
let send_loop = tokio::task::spawn(async move { /* Socket output */ });
```

This allows:
- **Non-blocking socket I/O** (recv/send loops)
- **Blocking computation** in process loop
- **Independent timing** for each component

### 4. **GPU Memory Management**
The system handles GPU memory through:

```rust
// Voice conditioning can load custom voice files
let ca_src = self.voice_ca_src(query.voice.as_ref(), query.voices.as_ref())?;

// Model weights shared but state is per-request
let mut state = moshi::tts_streaming::State::new(...);

// Audio tokenizer reset per request
let mut audio_tokenizer = self.audio_tokenizer.clone();
audio_tokenizer.reset_state();
```

## Key Differences from ASR Batched System

| Aspect | ASR System | TTS System |
|--------|------------|------------|
| **Concurrency** | Batched parallel processing | Sequential with mutex |
| **GPU Utilization** | Fixed-size batch tensors | Single request tensors |
| **State Management** | Slot-based client states | Per-request independent state |
| **Throughput** | High (batched) | Low (sequential) |
| **Latency** | Higher per request | Lower per request |

## Request Flow

1. **Request Arrives**: Mutex ensures only one runs
2. **State Initialization**: Creates independent TTS generation state
3. **Voice Conditioning**: Loads/encodes speaker voice if needed
4. **Generation Loop**: Processes tokens one by one in sequence
5. **Audio Decoding**: Converts tokens to PCM audio chunks
6. **Streaming Output**: Sends audio chunks back via WebSocket

## Performance Characteristics

- **GPU Underutilization**: Single requests don't fully utilize GPU capacity
- **Higher Latency for Concurrent Users**: Requests queue behind each other
- **Consistent Per-Request Performance**: No interference between requests
- **Simpler Error Handling**: Failures are isolated to single requests

The design prioritizes **simplicity and quality** over throughput - typical for TTS systems where generation quality is more critical than handling many concurrent users.

# TTS batch size meaning

Looking at the code more carefully, I need to correct my analysis. The `batch_size` parameter in the TTS config likely refers to **internal model batching** rather than concurrent request handling.

## What the `batch_size` Parameter Actually Controls

The `batch_size` parameter in TTS configuration most likely controls:

### 1. **Internal Model Batch Processing**
Even though requests are processed sequentially, the underlying models (LM and Mimi) may still use batching internally for efficiency:

```rust
// In the model initialization, we see references to batched operations
let mut audio_tokenizer = self.audio_tokenizer.clone();
audio_tokenizer.reset_state();
// The tokenizer is likely configured for a specific batch size
```

### 2. **Voice Conditioning Batching**
When processing multiple voices or speakers:

```rust
pub fn voice_ca_src(
    &self,
    voice: Option<&String>,
    voices: Option<&Vec<String>>,  // Multiple voices processed together
) -> Result<Tensor> {
    // When voices is Some(Vec), multiple voice embeddings are batched together
    for voice in voices.iter() {
        // Process each voice...
    }
    Ok(self.speaker_encoder.encode(&pcms)?)  // Batch encoding
}
```

### 3. **Tensor Operations Batch Dimensions**
The batch size affects how tensors are shaped and processed:

```rust
// Operations likely use the configured batch size for tensor dimensions
let audio_tokens = candle::Tensor::from_vec(audio_tokens, (1, cb, 1), state.device())?;
// The first dimension (1) would be batch_size in batched operations
```

## Real Purpose of `batch_size` in TTS Config

The `batch_size` parameter is used for:
- **Configuring model internal states** to expect certain batch dimensions
- **Voice embedding batching** when processing multiple speakers
- **Memory allocation planning** for GPU tensors
- **Consistency with shared model infrastructure** that expects batched operations

## Why Requests Are Still Sequential

The sequential processing (mutex) is separate from the internal batch size configuration because:
- **TTS Quality**: Real-time TTS often requires consistent, low-latency responses per user
- **Voice Personalization**: Each request may use different voice conditioning
- **State Management Complexity**: Managing multiple concurrent generation states is complex

## Example Usage

```yaml
tts:
  batch_size: 4  # Internal model batch size
  # This allows the model to handle batched tensor operations efficiently
  # but concurrent HTTP/WebSocket requests are still processed one at a time
```

So the `batch_size` parameter optimizes the **internal computation efficiency** rather than enabling **concurrent request processing**. The actual request concurrency is controlled separately by the server's request handling architecture (the mutex in this case).

# Mimi and lm

Looking at `mimi.rs` and `lm.rs`, I can now see how they integrate into the multi-user architecture. The key insight is that these modules handle **real-time streaming** differently from the batched ASR system.

## Multi-User Architecture with GPU Batching

### 1. **Room-Based Broadcasting System**

The `mimi.rs` module implements a **room-based broadcasting** architecture:

```rust
struct Room {
    sender: Arc<tokio::sync::Mutex<Sender>>,
    header_message: ws::Message,
    rx: tokio::sync::broadcast::Receiver<ws::Message>,
}

struct Mimi {
    rooms: std::collections::HashMap<String, Room>,
    // ... other fields
}
```

This allows:
- **Multiple users per room** receiving the same audio stream
- **Single sender per room** broadcasting to all receivers
- **Broadcast channel** for efficient distribution

### 2. **Real-Time Streaming vs Batched Processing**

Unlike the batched ASR system, `mimi.rs` and `lm.rs` are designed for **real-time streaming**:

```rust
// In mimi.rs - processing individual audio chunks as they arrive
Ok(MsgType::Codes) => {
    let codes: Vec<u32> = msg[1..]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let ncodes = codes.len();
    let codes = Tensor::from_vec(codes, (1, ncodes, 1), &self.device)?;  // Batch size = 1
    let pcm = audio_tokenizer.decode_step(&codes.into(), &().into())?;
    // ... process individual PCM chunks
}
```

### 3. **GPU Batch Size Integration**

The `batch_size` parameter affects these modules through:

**Tensor Shape Configuration:**
```rust
// The tensor is shaped with batch dimension = 1 for real-time processing
let codes = Tensor::from_vec(codes, (1, ncodes, 1), &self.device)?;
// But the underlying model is configured for batch_size during initialization
let audio_tokenizer = moshi::mimi::load(&mimi.audio_tokenizer_file, Some(8), dev)?;
```

**Model Initialization:**
```rust
// In both mimi.rs and lm.rs - models are loaded with batch size awareness
let audio_tokenizer = moshi::mimi::load(&lm.audio_tokenizer_file, Some(8), dev)?;
```

### 4. **Multi-User Integration Flow**

Here's how multi-user requests integrate:

**Audio Reception (`mimi.rs`):**
1. **Single sender** connects and sends audio codes
2. **Real-time decoding** happens with batch_size=1 tensors
3. **Broadcasting** to all room participants
4. **Multiple receivers** get the same decoded audio stream

**Language Modeling (`lm.rs`):**
1. **Single WebSocket** connection processes incoming audio
2. **Real-time token generation** with batch_size=1 operations
3. **Text and audio responses** sent back over WebSocket
4. **Broadcasting** text transcriptions/audio to connected clients

### 5. **Why Not Full Batching?**

These modules don't use full batching because:

```rust
// Real-time constraints - can't wait to fill a batch
while let Some(opus) = opus_in_rx.recv().await {
    // Process immediately as chunks arrive
    let pcm = Tensor::new(pcm, &dev)?.reshape((1, 1, ()))?;  // Immediate processing
    // ... immediate token generation
}
```

**Reasons:**
- **Latency requirements** - users expect real-time responses
- **Variable arrival times** - users don't send data synchronously
- **Individual user contexts** - each user may have different state

### 6. **Batch Size Parameter Purpose**

The `batch_size` in config serves to:
- **Pre-allocate GPU memory** for tensor operations
- **Configure model internal states** to handle batched operations efficiently
- **Maintain consistency** with shared model infrastructure
- **Allow future batching optimizations** without model reconfiguration

## Architecture Summary

```
Multiple Users → Individual WebSocket Connections
     ↓
Real-time Processing (batch_size=1 per user chunk)
     ↓
Shared Model Weights (configured for batch_size=N)
     ↓
Room-based Broadcasting (for distribution to multiple listeners)
     ↓
Individual User Responses
```

The system optimizes for **real-time interactive experience** rather than batch efficiency, using the batch_size parameter to configure underlying models for optimal performance while processing individual user requests.

# Clients refused

Based on the code analysis, the Rust server refuses clients based on **fixed capacity limits** derived from configuration parameters, not on dynamic load. Here's exactly when and why clients are refused:

## 1. **ASR Service - Fixed Batch Size Limits**

In `batched_asr.py`, clients are refused when:
```python
# When all batch slots are occupied
reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
# If no available slots, client request cannot be queued (TODO mentions this)
```

**Refusal Point**: When all `batch_size` slots are in use and no channel is available:
- **Config Parameter**: `batch_size` in the ASR configuration
- **Mechanism**: `channels()` method returns `None` when no slots available
- **Client Experience**: Immediate refusal, no queuing (though TODO suggests adding batch queuing)

## 2. **TTS Service - Mutex-Based Single Request**

In `src/tts.rs`, clients are refused when:
```rust
pub async fn handle_socket(&self, socket: ws::WebSocket, query: crate::TtsStreamingQuery) -> Result<()> {
    let _guard = self.mutex.lock().await;  // Only one request at a time
    // ... processing
}
```

**Refusal Point**: When attempting to acquire the mutex:
- **Config Parameter**: Implicit in the mutex design (always 1)
- **Mechanism**: Tokio's mutex timing out or connection queue limits
- **Client Experience**: Connection may timeout or be rejected at network level

## 3. **MIMI Service - Room Capacity Limits**

In `src/mimi.rs`, clients are refused when:
```rust
// Only one producer per room allowed
let mut sender = match room.sender.try_lock() {
    Ok(s) => s,
    Err(_) => anyhow::bail!("already a producer"),  // Refused immediately
};
```

**Refusal Points**:
1. **Producer Limit**: Only one sender per room
2. **Room Existence**: Non-existent room IDs
3. **Broadcast Channel**: Channel capacity limits (10 messages buffered)

## 4. **LM Service - Implicit Connection Limits**

In `src/lm.rs`, clients face implicit limits:
```rust
// No explicit queuing or capacity management
// Each WebSocket connection gets its own processing
let (opus_in_tx, mut opus_in_rx) = tokio::sync::mpsc::unbounded_channel();
```

**Refusal Points**:
- **System Resources**: OS-level connection limits
- **Memory Exhaustion**: No explicit bounds checking
- **Timeouts**: Fixed 360-second timeout

## Key Refusal Patterns

### **Hard Refusals** (Immediate):
- **TTS**: When mutex is held by another client
- **MIMI**: When trying to become producer in occupied room
- **ASR**: When all batch slots are occupied (no queuing)

### **Soft Refusals** (Timeout/Resource):
- **All Services**: When system resources exhausted
- **All Services**: When network timeouts occur
- **MIMI/LM**: When broadcast channels overflow

## Configuration Parameters That Control Capacity

```yaml
# These parameters directly control client capacity:
asr:
  batch_size: 8        # Max concurrent ASR requests

tts:
  batch_size: 1        # Internal batch size (concurrency still = 1 due to mutex)

mimi:
  rooms: ["room1", "room2"]  # Fixed number of rooms
  # Each room allows exactly 1 producer + unlimited consumers
```

## No Dynamic Load Management

The server notably **lacks** dynamic load management:
- **No queueing systems** (except implicit OS-level connection queues)
- **No resource monitoring** or adaptive refusal
- **No load-based scaling** decisions
- **Fixed resource allocation** based on configuration

## Client Experience

1. **ASR Clients**: Refused when all `batch_size` slots full
2. **TTS Clients**: Blocked until previous client finishes (de facto refusal under load)
3. **MIMI Clients**: Refused immediately if room producer slot taken
4. **LM Clients**: May timeout or be refused at system level under high load

The system is designed for **predictable, fixed-capacity operation** rather than elastic scaling, with refusal decisions based purely on **pre-configured limits** rather than dynamic system load assessment.

# TTS token generation logic

Now I can provide a detailed explanation of the `step` method in `tts_streaming.rs`. This is where the language model generates output tokens.

## The `step` Method Breakdown

The `step` method in `State` is the core of token generation:

```rust
pub fn step(
    &mut self,
    prev_text_token: u32,
    allowed_tokens: AllowedTokens,
    conditions: Option<&crate::conditioner::Condition>,
) -> Result<u32> {
```

### 1. **Audio Code Preparation**
First, it prepares the audio codes (tokens) for the model input:

```rust
let mut codes = Vec::with_capacity(self.model.generated_audio_codebooks());
let dev = self.model.device();
let batch_size = if self.cfg_alpha.is_some() { 2 } else { 1 };

for codebook in 0..self.model.generated_audio_codebooks() {
    // Complex logic to determine the correct audio token for each codebook
    // Takes into account delays, padding, and previous tokens
    let t = if codebook == 0 {
        if self.step_idx == 0 {
            Some(self.audio_pad_token())
        } else if self.step_idx <= self.config.text_audio_delay_in_tokens {
            None // Use literal zeros for first few seconds
        } else {
            Some(self.audio_tokens[self.step_idx - 1][codebook])
        }
    } else if self.step_idx <= self.config.acoustic_delay {
        Some(self.audio_pad_token())
    } else if self.step_idx <= self.config.text_audio_delay_in_tokens + self.config.acoustic_delay {
        None // Use literal zeros
    } else {
        Some(self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1][codebook])
    };
    
    // Convert to tensor for model input
    let t = match t {
        Some(t) => Some(Tensor::from_vec(vec![t; batch_size], (batch_size, 1), dev)?),
        None => None,
    };
    codes.push(t)
}
```

### 2. **Previous Text Token Preparation**
```rust
let prev_text_token =
    Some(Tensor::from_vec(vec![prev_text_token; batch_size], (batch_size, 1), dev)?);
```

### 3. **Language Model Forward Pass**
This is where the actual language model inference happens:

```rust
let (text_logits, ys) = match self.ca_src.as_ref() {
    None => self.model.forward_cond(prev_text_token, codes, conditions, &().into())?,
    Some(ca_src) => {
        self.model.forward_ca(prev_text_token, codes, ca_src, conditions, &().into())?
    }
};
```

**Key Points:**
- `forward_cond` or `forward_ca` calls the transformer model
- `ca_src` provides speaker conditioning information
- `conditions` provides additional control conditions
- Returns `text_logits` (for text token prediction) and `ys` (for audio token generation)

### 4. **Classifier-Free Guidance (CFG) Adjustment**
```rust
let text_logits = match self.cfg_alpha {
    None => text_logits.i((0, 0))?,
    Some(a) => match text_logits.dim(0)? {
        2 => ((text_logits.i((0, 0))? * a)? - (text_logits.i((1, 0))? * (a - 1.))?)?,
        b_size => candle::bail!("unexpected batch size {b_size}"),
    },
};
```

### 5. **Text Token Sampling**
Based on allowed tokens and sampling strategy:

```rust
let text_token = match allowed_tokens {
    AllowedTokens::Text(v) => v,           // Forced token
    AllowedTokens::Pad => self.config.text_pad_token,  // Forced pad
    AllowedTokens::PadOrEpad => {          // Sample between pad or end-of-phrase
        if self.consecutive_pads > self.config.max_consecutive_pads {
            self.config.text_eop_token
        } else {
            let text_token = self.text_lp.sample(&text_logits)?;
            if text_token == self.config.text_pad_token {
                self.config.text_pad_token
            } else {
                self.config.text_eop_token
            }
        }
    }
};
```

### 6. **Consecutive Pad Tracking**
```rust
if text_token == self.config.text_pad_token {
    self.consecutive_pads += 1
} else {
    self.consecutive_pads = 0
}
self.text_tokens[self.step_idx] = text_token;
```

### 7. **Audio Token Generation**
Uses the model's depformer to generate audio tokens:

```rust
let last_audio_tokens = if self.step_idx < self.config.text_audio_delay_in_tokens {
    None
} else {
    match self.cfg_alpha {
        None => self.model.depformer_sample(
            &ys,
            Some(text_token),
            self.forced_audio_tokens.forced_tokens(self.step_idx),
            &mut self.audio_lp,
        )?,
        Some(cfg_alpha) => self.model.depformer_sample_cfg(
            &ys,
            cfg_alpha,
            Some(text_token),
            self.forced_audio_tokens.forced_tokens(self.step_idx),
            &mut self.audio_lp,
        )?,
    }
};
```

### 8. **Audio Token Storage**
Stores generated audio tokens with proper delay handling:

```rust
let audio_pad_token = self.audio_pad_token();
for c_idx in 0..self.model.generated_audio_codebooks() {
    let delay = if c_idx == 0 { 0 } else { self.config.acoustic_delay };
    let pos = &mut self.audio_tokens[self.step_idx.saturating_sub(delay)][c_idx];
    match last_audio_tokens.as_ref() {
        Some(lat) => {
            if *pos == UNGENERATED {
                *pos = lat[c_idx]
            }
        }
        None => {
            if *pos == UNGENERATED {
                *pos = audio_pad_token
            }
        }
    }
}
```

### 9. **State Updates and Return**
```rust
self.step_idx += 1;
if self.step_idx >= self.audio_tokens.len() {
    candle::bail!("max step-idx reached")
}
Ok(text_token)
```

## Key Architecture Insights

1. **Interleaved Generation**: Text and audio tokens are generated together in a coordinated fashion
2. **Delay Handling**: Complex delay management for proper audio-text synchronization
3. **Conditioning**: Speaker embedding (`ca_src`) and control conditions influence generation
4. **Classifier-Free Guidance**: Optional CFG for improved generation quality
5. **Token Constraints**: `AllowedTokens` enforces generation constraints based on input state

This method is called repeatedly in the generation loop in `tts.rs`, with each call advancing the generation by one step and returning the newly generated text token.

# Batch transformer processing

Looking at `lm_generate_multistream.rs`, I can explain how the transformer batch size is managed:

## Batch Size Management in Transformer Operations

### 1. **Dynamic Batch Size Based on CFG**
The batch size is determined dynamically based on whether Classifier-Free Guidance (CFG) is used:

```rust
let batch_size = if self.cfg_alpha.is_some() { 2 } else { 1 };
```

- **Batch Size = 1**: Normal operation without CFG
- **Batch Size = 2**: When CFG is enabled (one for conditional, one for unconditional)

### 2. **Tensor Creation with Dynamic Batch Size**
All tensors are created with the dynamic batch size:

```rust
// Audio tokens tensor
let t = Tensor::from_vec(vec![t; batch_size], (batch_size, 1), dev)?;

// Text token tensor  
let text_token = match text_token {
    Some(text_token) => {
        Some(Tensor::from_vec(vec![text_token; batch_size], (batch_size, 1), dev)?)
    }
    None => None,
};
```

### 3. **Model Forward Pass with Batch Support**
The model handles the batched inputs:

```rust
let (text_logits, ys) = match ca_src.as_ref() {
    None => {
        let (logits, ys) = self.model.forward_cond(text_token, codes, conditions, &().into())?;
        let logits = match self.cfg_alpha {
            None => logits.i((0, 0))?,  // Extract first batch element
            Some(a) => match logits.dim(0)? {
                2 => ((logits.i((0, 0))? * a)? - (logits.i((1, 0))? * (a - 1.))?)?,  // CFG combination
                b_size => candle::bail!("unexpected batch size {b_size}"),
            },
        };
        (logits, ys)
    }
    Some(ca_src) => {
        if self.cfg_alpha.is_some() {
            candle::bail!("cfg is not supported with cross attention")
        }
        let (logits, ys) = self.model.forward_ca(text_token, codes, ca_src, None, &().into())?;
        (logits.i((0, 0))?, ys)  // Extract first batch element
    }
};
```

### 4. **Batch Size Handling in Audio Generation**
The audio token generation also respects the batch size:

```rust
let last_audio_tokens = match self.cfg_alpha {
    None => self.model.depformer_sample(
        &ys,
        Some(text_token),
        self.forced_audio_tokens.forced_tokens(self.step_idx),
        &mut self.audio_lp,
    )?,
    Some(cfg_alpha) => self.model.depformer_sample_cfg(
        &ys,
        cfg_alpha,
        Some(text_token),
        self.forced_audio_tokens.forced_tokens(self.step_idx),
        &mut self.audio_lp,
    )?,
};
```

### 5. **CFG Logic for Batch Processing**
When CFG is enabled, the system:
1. Processes both conditional and unconditional paths in a single batch of size 2
2. Combines the results using the alpha parameter:

```rust
// CFG combination: conditional * alpha - unconditional * (alpha - 1)
Some(a) => match logits.dim(0)? {
    2 => ((logits.i((0, 0))? * a)? - (logits.i((1, 0))? * (a - 1.))?)?,
    b_size => candle::bail!("unexpected batch size {b_size}"),
},
```

## Key Insights

1. **Runtime Batch Size Determination**: Batch size is determined at inference time based on CFG usage
2. **Model Compatibility**: The underlying transformer model must support batched operations
3. **Memory Efficiency**: Only uses batch size 2 when necessary (CFG enabled)
4. **Constraint Handling**: CFG is incompatible with cross-attention (`ca_src`) due to complexity

The batch size management is minimal but effective - it scales between 1 and 2 based on whether advanced sampling techniques like CFG are enabled, allowing the transformer to efficiently handle both simple and complex generation scenarios.

`batched_transformer.rs` implements a **streaming transformer architecture** designed for efficient batched processing. Here's how it works:

## Core Architecture

### 1. **Batched KV Cache Management**
The key innovation is the **KV Cache Builder** that pre-allocates memory for batched operations:

```rust
pub struct StreamingTransformer {
    builder: KvCacheBuilder,  // Pre-allocated batched KV cache
    layers: Vec<StreamingTransformerLayer>,
}

impl StreamingTransformer {
    pub fn new(batch_size: usize, cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let builder = KvCacheBuilder::new(batch_size, cfg.context, vb.dtype(), vb.device())?;
        // ... initialize layers with shared builder
    }
}
```

The `KvCacheBuilder` creates fixed-size caches for all batch elements, eliminating dynamic allocation during inference.

### 2. **Streaming Multihead Attention**
The attention mechanism handles streaming inputs with KV caching:

```rust
impl StreamingMultiheadAttention {
    pub fn forward(&mut self, xs: &Tensor, rope: Option<&Rope>, iam: &IndicesAndMask) -> Result<Tensor> {
        // Project input to QKV
        let qkv = xs.apply(&self.in_proj)?.reshape((b, t, 3, self.num_heads, head_dim))?;
        
        // Apply rotary embeddings if needed
        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;
        
        if let Some(rope) = rope.as_ref() {
            q = rope.apply_rotary_emb(&q)?;
            k = rope.apply_rotary_emb(&k)?;
        }

        // Append to KV cache and retrieve cached + new keys/values
        let (k, v) = { self.kv_cache.append(&k.contiguous()?, &v.contiguous()?, iam)? };
        
        // Perform attention computation with context limiting
        let k_len = k.dim(2)?;
        let k_target_len = t + usize::min(self.context, k_len - t);
        let (k, v) = if k_target_len < k_len {
            let k = k.narrow(2, k_len - k_target_len, k_target_len)?;
            let v = v.narrow(2, k_len - k_target_len, k_target_len)?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        // Standard scaled dot-product attention
        let xs = {
            let pre_ws = q.matmul(&k.t()?)?;
            let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;
            let pre_ws = pre_ws.broadcast_add(iam.mask())?;
            let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?;
            ws.matmul(&v)?
        };
        
        xs.apply(&self.out_proj)
    }
}
```

### 3. **Batched Mask Handling**
The system uses `IndicesAndMask` to efficiently handle batched attention masks:

```rust
let iam = match m.cpu() {
    None => candle::bail!("batched-transformer expects a mask"),
    Some(m) => self.builder.indices_and_mask(t, m)?,
};
```

This allows different batch elements to have different attention patterns while maintaining computational efficiency.

### 4. **Streaming Processing Interface**
The transformer implements a streaming interface for incremental processing:

```rust
impl StreamingModule for StreamingTransformer {
    fn reset_state(&mut self) {
        self.builder.reset();  // Reset all batch KV caches
    }

    fn step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.forward(xs, m)?)),
        }
    }
}
```

## Key Features

### **Batched Memory Management**
- Pre-allocated KV caches for all batch elements
- Efficient memory reuse across time steps
- Context window management to limit memory usage

### **Flexible Positional Embeddings**
Support for multiple positional embedding types:
```rust
let rope = match cfg.positional_embedding {
    PositionalEmbedding::Rope => Some(RotaryEmbedding::new(...)?),
    PositionalEmbedding::None | PositionalEmbedding::Sin => None,
};
```

### **Cross-Attention Support**
Optional cross-attention layers with shared parameters:
```rust
let cross_attn = match cfg.cross_attention.map(|v| v.1) {
    Some(norm_type) => {
        let norm_cross = Norm::new_shortcut(d_model, norm_type, vb.pp("norm_cross"))?;
        let cross_attn = match shared_ca_vb {
            None => StreamingMultiheadCrossAttention::new(cfg, vb.pp("cross_attention"), None)?,
            Some(shared_vb) => StreamingMultiheadCrossAttention::new(
                cfg, shared_vb.pp("cross_attention"), Some(vb.pp("cross_attention.gate")),
            )?,
        };
        Some((norm_cross, cross_attn))
    }
    None => None,
};
```

### **Projection Layers**
Input/output projections for dimension matching:
```rust
pub struct ProjectedTransformer {
    transformer: StreamingTransformer,
    input_proj: Option<MaybeQuantizedLinear>,    // Project input to model dim
    output_projs: Vec<Option<MaybeQuantizedLinear>>, // Project outputs to target dims
}
```

## Batch Processing Workflow

1. **Initialization**: Create transformer with fixed `batch_size`
2. **Batched Forward Pass**: Process batch of sequences simultaneously
3. **KV Cache Updates**: Append new keys/values to batched caches
4. **Context Management**: Trim old cache entries beyond context window
5. **Streaming Steps**: Process one time step at a time for real-time applications
6. **Batch Reset**: Reset specific batch indices when sequences complete

## Performance Benefits

- **Memory Efficiency**: Pre-allocated batched caches eliminate fragmentation
- **Computational Efficiency**: Batched tensor operations maximize GPU utilization
- **Streaming Capability**: Incremental processing for real-time applications
- **Context Management**: Automatic handling of long sequences with sliding windows

This architecture is designed for scenarios where you need to process multiple sequences efficiently, either in batch mode for throughput or in streaming mode for low-latency applications.