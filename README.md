# Home Assistant Helper/Mocker (HAMock)

A Python utility for interacting with Home Assistant, primarily focused on state monitoring, logging, event replay capabilities, and security risk assessment through AI integration.

## Features

- [x] Listen to real-time state changes via WebSocket
- [x] Fetch current states of all entities
- [x] Dump state changes and entity states to files
- [x] Event replay functionality with configurable speed
- [x] Forward events to OpenAI-compatible endpoints (e.g., Ollama) for analysis
- [x] Security risk assessment through AI integration
- [x] Support Base64 encoded images in replay
- [x] Vision capabilities to leveraging multi-modal inference models (default llama3.2-vision)
- [ ] Mock the Home Assistant Endpoints to act as a proxy
- [ ] Use Ollama structured output


## Installation

Install dependencies:

```bash 
pip install -r requirements.txt
```

## Usage

Most updated documentation is probably in the help description of the utility:

```bash
./hamock.py -h
```

### Commands

#### Listen for state changes:
```bash
./hamock.py listen [--output OUTPUT_FILE] [--max-retries RETRIES] [--retry-delay DELAY] [--append]
```
The `--append` flag allows adding new events to an existing output file instead of overwriting it.

#### Get current states:
```bash
./hamock.py states [--output OUTPUT_FILE] [--timeout TIMEOUT]
```

#### Replay events:

```bash
./hamock.py replay --input INPUT_FILE [--replay-speed SPEED] 
```

#### Replay events with AI analysis:

```bash
./hamock.py replay --input INPUT_FILE --infer true [--replay-speed SPEED] \
                   [--openai-url URL] [--openai-model MODEL] \
                   [--model-seed SEED] [--stats] \
                   [--system-prompt PROMPT] 
```

The `--stats` flag enables displaying detailed inference statistics for each event processed. In inference mode (`--infer`), the default system prompt attempts to emulate a rudimentary home security system, and is defined as:

```
You are a home security system that monitors changes in the state of various sensors and entities. 
Your task is to assess each change for potential security risks and respond ONLY using a JSON structure 
in the format: { "security_risk": "LOW", "MEDIUM", or "HIGH" }. When assessing security risk, 
follow these guidelines:
- If the change involves an entity related to regular system operations or minor fluctuations 
(e.g., load changes, memory usage), mark it as "LOW".
- If the change involves an unexpected or unusual sensor state that might require attention but is not 
directly indicative of a critical problem, mark it as "MEDIUM".
- If the change indicates a potentially dangerous or highly unusual event that could indicate a security 
threat (e.g., unauthorized access attempts, sudden large fluctuations, loss of sensor communication), 
mark it as "HIGH".
Your goal is to provide a reasonable evaluation based on these guidelines. Always keep your output strictly 
in the format: { "security_risk": "LOW", "MEDIUM", or "HIGH" }.
```

If the model is not available but the inference endpoint is, the application will wait 30s in a loop in an attempt to allow the orchestrator to download/load the model.

### Environment variables and Global options

By default, the following environment variables are used:

- `HAMOCK_HASS_URL`: Home Assistant instance URL (default: `localhost:8123`)
- `HAMOCK_HASS_ACCESS_TOKEN`: Your Home Assistant Long-Lived Access Token ([check the documentation](https://community.home-assistant.io/t/how-to-get-long-lived-access-token/162159))
- `HAMOCK_OPENAI_URL`: OpenAI-compatible endpoint URL (default: `http://localhost:11434`)
- `HAMOCK_OPENAI_MODEL`: Model to use for inference (default: `llama3.2:1b-instruct-q4_K_M`)
- `HAMOCK_MODEL_SEED`: Seed for model inference (default: `36424`)
- `HAMOCK_SYSTEM_PROMPT`: System prompt for AI analysis (see above)
- `HAMOCK_REPLAY_SPEED`: Default replay speed multiplier (default: `1.0`)
- `HAMOCK_TIMEOUT`: Default timeout in seconds (default: `10`)
- `HAMOCK_RETRY_DELAY`: Default retry delay in seconds (default: `5`)
- `HAMOCK_MAX_RETRIES`: Default maximum retry attempts (default: `3`)
- `HAMOCK_INFER`: Enable inference during replay (default: `false`)
- `HAMOCK_STATS`: Display inference statistics (default: `false`)

These can be overridden by passing corresponding command-line arguments.

## Development Status

### Completed
- [x] WebSocket connection for state change monitoring
- [x] HTTP API integration for full state retrieval
- [x] File dump functionality
- [x] CLI help messages
- [x] Environment variable configuration
- [x] Event replay functionality
- [x] Configurable replay speed
- [x] OpenAI/Ollama integration for event analysis
- [x] Security risk assessment through AI

### In Progress
- [ ] Capture still images from Home Assistant
- [x] Vision capabilities during inference
- [ ] Home Assistant endpoint mocking/proxy

## License

This project's source code is confidential with restricted distribution.
