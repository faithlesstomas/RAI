# Antigravity Localharness Implementation & Reverse Engineering Guide

This document provides a complete specification of the `localharness` communication protocol and outlines how to implement a custom replacement harness compatible with the Google Antigravity SDK.

---

## 1. Architectural Role & Execution Flow

The `localharness` is a Go-based subprocess executed by the Python Antigravity SDK (`LocalConnectionStrategy`). It functions as an agent orchestration gateway, managing:
1. **The ReAct Execution Loop**: Direct coordination with LLMs (Gemini or local endpoints like Ollama).
2. **Context Window Compaction**: Compressing history when token limits are reached.
3. **OS Tool Execution (Harness-side)**: Executing built-in tools (`run_command`, `file_edit`, etc.) locally.
4. **Tool Intermediation (Client-side)**: Marshaling custom python tool definitions, sending tool call requests to the Python client, and awaiting execution results.
5. **Persistence**: Storing and loading session history via binary Trajectory protobuf files (`traj-{conversation_id}`).

---

## 2. Startup Handshake Protocol

When Python initializes an agent, it executes the `localharness` binary as a subprocess and performs a length-prefixed protobuf handshake over standard input/output streams.

```
┌──────────────┐                        ┌──────────────┐
│  Python SDK  │                        │ localharness │
└──────┬───────┘                        └──────┬───────┘
       │                                       │
       │  1. Write InputConfig (stdin)         │
       │ ── ── ── ── ── ── ── ── ── ── ── ── > │
       │                                       │
       │  2. Read OutputConfig (stdout)        │
       │ < ── ── ── ── ── ── ── ── ── ── ── ── │
       │                                       │
       │  3. Connect WebSocket (port + key)    │
       │ ── ── ── ── ── ── ── ── ── ── ── ── > │
       │                                       │
       │  4. Send InitializeConversationEvent  │
       │ ── ── ── ── ── ── ── ── ── ── ── ── > │
```

### 2.1 Handshake Serialization Detail
All messages on `stdin`/`stdout` are framed using a **4-byte Little-Endian unsigned integer** indicating the size of the following serialized Protobuf message.
```
[ 4 Bytes: Length (uint32) ] [ N Bytes: Serialized Protobuf Payload ]
```

### 2.2 Phase 1: InputConfig (Python -> Go `stdin`)
Python sends the following configuration:
```protobuf
message InputConfig {
  optional string storage_directory = 1; // Dir where trajectories are stored
  optional uint32 port = 2;              // Requested port (0 for auto)
  optional string bind_address = 3 [default = "localhost"];
}
```

### 2.3 Phase 2: OutputConfig (Go -> Python `stdout`)
The Go harness starts a WebSocket server and writes its listening parameters back:
```protobuf
message OutputConfig {
  optional int32 port = 1;       // The port the WebSocket server is listening on
  optional string api_key = 2;   // A randomly generated security token for connection authentication
}
```

---

## 3. WebSocket Channel & Message Schema

Once the WebSocket port is known, Python connects to `ws://{bind_address}:{port}/` sending the `x-goog-api-key: {api_key}` header.

### 3.1 Session Initialization
The client immediately sends an `InitializeConversationEvent`:
```protobuf
message InitializeConversationEvent {
  optional HarnessConfig config = 1;
}

message HarnessConfig {
  optional string cascade_id = 1;                    // The conversation ID (resumption key)
  optional GeminiConfig gemini_config = 2;           // Gemini Model details
  optional GemmaConfig gemma_config = 3;             // Local Ollama/Gemma Model details (OneOf with gemini_config)
  optional SystemInstructions system_instructions = 4;
  repeated Tool tools = 5;                           // Custom Python tool schemas passed from the SDK
  optional HarnessSideTools harness_side_tools = 6;  // Exposes/hides built-in tools (view_file, run_command, etc.)
  optional uint32 compaction_threshold = 7;
  repeated Workspace workspaces = 8;                 // Active secure directory bounds
  repeated string skills_paths = 9;
  optional string finish_tool_schema_json = 10;
  optional bytes initial_trajectory = 11;
  optional string app_data_dir = 12;
}
```

If `cascade_id` matches an existing trajectory file (`traj-{cascade_id}`) in `storage_directory`, the harness loads it and replays the past steps over the WebSocket before signaling it is ready.

---

## 4. The Message Exchange Event Loop

All subsequent messaging follows the `InputEvent` (client -> harness) and `OutputEvent` (harness -> client) schemas.

```
                  ┌──────────────┐
                  │ InputEvent   │
                  └──────┬───────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
[ user_input ]    [ tool_response ]  [ tool_confirmation ]
(Text prompt)     (Python result)    (User security approval)
                         │
                         ▼
                  ┌──────────────┐
                  │ OutputEvent  │
                  └──────┬───────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
 [ step_update ]  [ trajectory_update ] [ tool_call ]
(Thoughts/deltas)  (Running/Idle state)  (Python callback req)
```

### 4.1 InputEvent (Client -> Harness)
```protobuf
message InputEvent {
  oneof event {
    string user_input = 1;
    UserInput complex_user_input = 7;      // Multipart/Multimedia inputs
    ToolConfirmation tool_confirmation = 2; // HITL decision
    ToolResponse tool_response = 3;         // Python tool result
    UserQuestionsResponse question_response = 4;
    bool halt_request = 5;
    string automated_trigger = 6;
  }
}
```

### 4.2 OutputEvent (Harness -> Client)
```protobuf
message OutputEvent {
  optional int64 seq_num = 1;
  optional int64 timestamp_micros = 2;
  
  oneof event {
    StepUpdate step_update = 10;
    TrajectoryStateUpdate trajectory_state_update = 11;
    ToolCall tool_call = 12; // Request Python client to run a custom tool
    UsageMetadata usage_metadata = 20;
  }
}
```

---

## 5. Anatomy of a Step Update (`StepUpdate`)

Each transition in the agentic loop is broadcast via `StepUpdate`. It maps whether the agent is thinking, outputting text, calling local tools, or waiting for user interaction.

```protobuf
message StepUpdate {
  optional string cascade_id = 1;
  optional string trajectory_id = 2;
  optional uint32 step_index = 3;
  optional State state = 4;
  optional Source source = 5;
  optional Target target = 6;
  optional string error_message = 7;
  optional string thinking = 8;
  optional string text_delta = 9;
  optional string thinking_delta = 10;
  optional string text = 20;

  // Active Tool Action payloads (OneOf)
  optional ActionListDirectory list_directory = 21;
  optional ActionFindFile find_file = 22;
  optional ActionSearchDirectory search_directory = 23;
  optional ActionViewFile view_file = 24;
  optional ActionCreateFile create_file = 25;
  optional ActionEditFile edit_file = 26;
  optional ActionRunCommand run_command = 27;
  optional ActionCompaction compaction = 28;
  optional ActionInvokeSubagent invoke_subagent = 29;
  optional ActionGenerateImage generate_image = 30;
  optional ActionFinish finish = 31;
  optional ActionError error = 32;

  // Human-in-the-Loop Requests
  optional string request_text = 50;
  optional ToolConfirmationRequest tool_confirmation_request = 51;
  optional UserQuestionsRequest questions_request = 52;

  enum State {
    STATE_UNSPECIFIED = 0;
    STATE_ACTIVE = 1;           // Running/Generating
    STATE_DONE = 2;             // Step finished successfully
    STATE_WAITING_FOR_USER = 3; // Suspended for HITL approval/input
    STATE_ERROR = 4;
  }

  enum Source {
    SOURCE_UNSPECIFIED = 0;
    SOURCE_SYSTEM = 1;
    SOURCE_USER = 2;
    SOURCE_MODEL = 3;
  }

  enum Target {
    TARGET_UNSPECIFIED = 0;
    TARGET_USER = 1;
    TARGET_MODEL = 2;
    TARGET_ENVIRONMENT = 3;
  }
}
```

---

## 6. How to Implement a Custom Harness

To build a custom `localharness` in Python or Go, follow these steps:

### Phase 1: Implement Handshake & WebSocket Server
1. Listen on `stdin` for a little-endian length prefix, followed by the `InputConfig` protobuf payload.
2. Bind a WebSocket server to the requested port/address.
3. Generate a cryptographically secure token (e.g. UUID) for `api_key`.
4. Serialize and write the `OutputConfig` containing the port and `api_key` to `stdout`, prefixed by its 4-byte length.
5. Await WebSocket connection and verify the `x-goog-api-key` header matches your `api_key`.

### Phase 2: Handle Initialization & Load History
1. Read the `InitializeConversationEvent` over the WebSocket.
2. Resolve `cascade_id`. Look in `storage_directory` for any trajectory state file matching `traj-{cascade_id}`.
3. If it exists:
   * Load the trajectory steps.
   * Send them sequentially as `OutputEvent(step_update=...)` messages with `state = STATE_DONE`.
   * Send `OutputEvent(trajectory_state_update=TrajectoryStateUpdate(state=STATE_IDLE))`.

### Phase 3: Implement ReAct & Tool Loops
1. Await an `InputEvent` containing `user_input`.
2. Send `OutputEvent(trajectory_state_update=TrajectoryStateUpdate(state=STATE_RUNNING))` to indicate execution has started.
3. Feed the prompt and history to the LLM (Gemini/Ollama).
4. Stream LLM output deltas back:
   * Parse thoughts and yield them via `thinking_delta` and `thinking`.
   * Parse conversational text and yield via `text_delta` and `text`.
5. If the LLM generates a tool call:
   * **If built-in (harness-side)**: Run it locally (e.g., execute the shell command, edit files using unified diffs) and return the result to the LLM context. Send a corresponding `StepUpdate` (e.g. `run_command` or `edit_file`) with status `STATE_DONE`.
   * **If Python (client-side)**: Send `OutputEvent(tool_call=ToolCall(...))` over the WebSocket. Suspend loop execution. Await `InputEvent` containing `tool_response`. Feed the result back to the LLM context and resume.
6. Once the model calls `finish` or halts, send `OutputEvent(trajectory_state_update=TrajectoryStateUpdate(state=STATE_IDLE))`.
7. Serialize the accumulated steps/history into protobuf format and write them to `storage_directory/traj-{cascade_id}`.
