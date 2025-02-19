#!/usr/bin/env python3

# The source code of this file is confidential, with restricted distribution.
# Author: Hugo Sereno Ferreira <hugo.ferreira@...>

# [x] Forward replay events to an OpenAI Endpoint (e.g. Ollama)
# [x] Select the model to use for the OpenAI Endpoint
# [x] Measure the inference time and other stats
# [x] Configure the system prompt to use for the OpenAI Endpoint
# [x] If the model is not available, retry after a while (wait for the orchestrator to start it)
# [x] Support Base64 Images
# [ ] Mock the Home Assistant Endpoints to act as a proxy
# [ ] Handle user interruptions gracefully
# [ ] Use Ollama structured output

import asyncio
import websockets
import json
import logging
import argparse
import os
import requests
import time
from datetime import datetime, timezone
from typing import Optional, TextIO, Any
from rich.traceback import install

# Merge conflict. Need to solve this later
# from config import *
# from replay_events import replay_events

import base64

install(show_locals=True)

# Configure logging (this should be enough for now)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# System prompt used for security risk assessment
# This should all be in config.py. Merge conflict.

SYSTEM_PROMPT = ("You are a home security system that monitors changes in the state of various sensors and entities. "
        "Your task is to assess each change for potential security risks and respond ONLY using a JSON structure "
        "in the format: { \"security_risk\": \"LOW\", \"MEDIUM\", or \"HIGH\" }. When assessing security risk, "
        "follow these guidelines:\n"
        "- If the change involves an entity related to regular system operations or minor fluctuations "
        "(e.g., load changes, memory usage), mark it as \"LOW\".\n" 
        "- If the change involves an unexpected or unusual sensor state that might require attention but is not "
        "directly indicative of a critical problem, mark it as \"MEDIUM\".\n"
        "- If the change indicates a potentially dangerous or highly unusual event that could indicate a security "
        "threat (e.g., unauthorized access attempts, sudden large fluctuations, loss of sensor communication), "
        "mark it as \"HIGH\".\n"
        "Your goal is to provide a reasonable evaluation based on these guidelines. Always keep your output strictly "
        "in the format: { \"security_risk\": \"LOW\", \"MEDIUM\", or \"HIGH\" }.")

# Configuration constants with environment fallbacks (Docker, I'm looking at you)
DEFAULT_URL = os.getenv("HAMOCK_HASS_URL", "localhost:8123")
DEFAULT_TOKEN = os.getenv("HAMOCK_HASS_ACCESS_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmN2NjNTkyY2VkMjI0NTY2OWI4ZDY0OWQ4MmFkOGFlYiIsImlhdCI6MTcyODQwMzg4OSwiZXhwIjoyMDQzNzYzODg5fQ.OqEiy5SJFdjc70CFWR9IiP9eXgI7aAY8N7YeowfvEtM")
DEFAULT_TIMEOUT = int(os.getenv("HAMOCK_TIMEOUT", "10"))
DEFAULT_RETRY_DELAY = int(os.getenv("HAMOCK_RETRY_DELAY", "5"))
DEFAULT_MAX_RETRIES = int(os.getenv("HAMOCK_MAX_RETRIES", "3"))
DEFAULT_REPLAY_SPEED = float(os.getenv("HAMOCK_REPLAY_SPEED", "1.0"))
DEFAULT_OPENAI_URL = os.getenv("HAMOCK_OPENAI_URL", "http://localhost:11434")
DEFAULT_OPENAI_MODEL = os.getenv("HAMOCK_OPENAI_MODEL", "llama3.2-vision:latest")
DEFAULT_MODEL_SEED = int(os.getenv("HAMOCK_MODEL_SEED", "123"))
DEFAULT_SYSTEM_PROMPT = os.getenv("HAMOCK_SYSTEM_PROMPT", SYSTEM_PROMPT)
DEFAULT_INFER = bool(os.getenv("HAMOCK_INFER", "false").lower() == "true")
DEFAULT_DISPLAY_STATS = bool(os.getenv("HAMOCK_DISPLAY_STATS", "false").lower() == "true")

# Home assistant state changed event transformer
# See https://developers.home-assistant.io/docs/api/websocket/#subscribe-to-events
def transform_state_changed_event(event: dict) -> dict:
    data = event["event"]["data"]
    timestamp = event["event"]["time_fired"]  # Home Assistant provides ISO format timestamp (I think...)
    
    transformed_event = {
        "timestamp": timestamp,
        "entity_id": data["entity_id"],
        "old_state": data["old_state"]["state"] if data["old_state"] else "unknown",
        "new_state": data["new_state"]["state"] if data["new_state"] else "unknown"
    }
    
    return transformed_event

# Home assistant listener
# See https://developers.home-assistant.io/docs/api/websocket/
async def listen_to_home_assistant(url: str = DEFAULT_URL, 
                                   token: str = DEFAULT_TOKEN, 
                                   dump_file: str | None = None, 
                                   max_retries: int = DEFAULT_MAX_RETRIES, 
                                   retry_delay: int = DEFAULT_RETRY_DELAY, 
                                   append: bool = False) -> None:
    while True:  # Connection retries
        try:
            async with websockets.connect(f"ws://{url}/api/websocket") as websocket:
                # Wait for the server to send "auth_required"
                auth_required = await websocket.recv()
                auth_data = json.loads(auth_required)
                if auth_data.get("type") != "auth_required":
                    logging.error("Unexpected initial message: %s", auth_data)
                    return

                # Authenticate
                await websocket.send(json.dumps({
                    "type": "auth",
                    "access_token": token
                }))

                # Wait for the authentication response
                auth_response = json.loads(await websocket.recv())
                if auth_response.get("type") != "auth_ok":
                    logging.error("Authentication failed: %s", auth_response)
                    return

                logging.info("Successfully authenticated")

                # Subscribe to state changes
                msg_id = 1
                await websocket.send(json.dumps({
                    "id": msg_id,
                    "type": "subscribe_events",
                    "event_type": "state_changed"
                }))

                # Wait for subscription confirmation
                sub_response = json.loads(await websocket.recv())
                if sub_response.get("success") is not True:
                    logging.error("Failed to subscribe: %s", sub_response)
                    return

                logging.info("Subscribed to state changes")
                logging.info("Press Ctrl+C to stop the listener")

                # Main event loop
                if dump_file:
                    with open(dump_file, "a" if append else "w", encoding="utf-8") as file_handle:
                        await process_events(websocket, file_handle)
                else:
                    await process_events(websocket, None)

        # Yeah... async code is fun
        except websockets.exceptions.ConnectionClosed as e:
            logging.error("WebSocket connection closed: %s", e)
            if max_retries > 0:
                logging.info(f"Retrying connection in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                max_retries -= 1
                continue
            break
        except Exception as e:
            logging.error("Unexpected error: %s", e)
            break

async def process_events(websocket: websockets.WebSocketClientProtocol, file_handle: Optional[TextIO]) -> None:
    try:
        while True:
            message = await websocket.recv()
            event = json.loads(message)

            if event.get("type") == "event" and event.get("event", {}).get("event_type") == "state_changed":
                transformed_event = transform_state_changed_event(event)
                
                log_message = (
                    f"Entity '{transformed_event['entity_id']}' changed state: "
                    f"{transformed_event['old_state']} -> {transformed_event['new_state']}"
                )
                logging.info(log_message)
                # Dump the message to the file if enabled

                if file_handle:
                    file_handle.write(json.dumps(transformed_event) + "\n")
                    file_handle.flush() # Eagerly write to disk

    except asyncio.CancelledError:
        logging.info("Shutting down listener...")
        raise

# Get the current states of the Home Assistant instance
# See https://developers.home-assistant.io/docs/api/rest/
def get_states(url: str = DEFAULT_URL, token: str = DEFAULT_TOKEN, dump_file: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> list[dict] | None:
    try:
        # Make the API request
        response = requests.get(
            f"http://{url}/api/states",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the JSON response
        states_data = response.json()
        logging.info(f"Received {len(response.text)} bytes from Home Assistant.")
        # Print the response as a JSON object, pretty-printed
        logging.info(json.dumps(states_data, indent=4))
        
        # Write to dump file if specified
        if dump_file:
            with open(dump_file, "w", encoding="utf-8") as f:
                json.dump(states_data, f)
                f.write("\n")
                
        return states_data
        
    # So many ways to fail...
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch states: {str(e)}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse response: {str(e)}")
    except IOError as e:
        logging.error(f"Failed to write to file {dump_file}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    
    return None

# This should all be in replay_events.py. Merge conflict.
async def replay_events(replay_file: str | None = None, replay_speed: float = DEFAULT_REPLAY_SPEED, openai_url: str = DEFAULT_OPENAI_URL, openai_model: str = DEFAULT_OPENAI_MODEL, model_seed: int = DEFAULT_MODEL_SEED, display_stats: bool = False, system_prompt: str = DEFAULT_SYSTEM_PROMPT, infer: bool = False) -> None:
    # Wish better types were a thing in Python
    # For future readers: yes, I like value-dependent types
    if not replay_file:
        logging.error("No replay file specified")
        return
    
    if replay_speed <= 0:
        logging.error("Replay speed must be positive")
        return
    
    if model_seed < 0:
        logging.error("Model seed must be non-negative")
        return
    
    # Test the OpenAI endpoint availability before proceeding
    if infer:
        if not test_openai_endpoint(openai_url, openai_model):
            return
        else:
            logging.info("OpenAI endpoint is available")
            logging.info(f"Using model: {openai_model}")
            logging.info(f"Using seed: {model_seed}")
            logging.info("--------------------------------")

    try:
        # Read and parse all events first
        events = []
        with open(replay_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    event = json.loads(line)
                    # Validate required fields
                    if not all(k in event for k in ["timestamp", "entity_id", "old_state", "new_state"]):
                        logging.warning(f"Line {line_num}: Missing required fields, skipping")
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    logging.warning(f"Line {line_num}: Malformed JSON, skipping: {line.strip()}")
                    continue

        if not events:
            logging.warning("No valid events found in replay file")
            return

        # Sort events by timestamp to ensure correct order
        # This is usually guaranteed by the dump of the log file... unless someone messes with it
        events.sort(key=lambda x: x["timestamp"])
        total_events = len(events)
        
        try:
            # Get the first event's timestamp as reference
            first_event_time = datetime.fromisoformat(events[0]["timestamp"].rstrip('Z')).replace(tzinfo=timezone.utc)
            last_event_time = None
            
            for idx, event in enumerate(events, 1):
                current_time = datetime.fromisoformat(event["timestamp"].rstrip('Z')).replace(tzinfo=timezone.utc)
                
                # Calculate and apply delay between events
                if last_event_time is not None:
                    time_diff = (current_time - last_event_time).total_seconds()
                    adjusted_delay = time_diff / replay_speed
                    if adjusted_delay > 0:
                        await asyncio.sleep(adjusted_delay)
                
                # Create human-readable output (signal if image data is present)
                log_message = (
                    f"[{event['timestamp']}] ({idx}/{total_events}) "
                    f"Entity '{event['entity_id']}' changed state: "
                    f"{event['old_state']} -> {event['new_state']}"
                    f"{' (image data present)' if 'image_data' in event else ''}"
                )
                logging.info(log_message)

                # Forward to OpenAI compatible Inference Endpoint
                if infer:
                    forward_to_openai(event, openai_url, openai_model, model_seed, display_stats, system_prompt)
                
                last_event_time = current_time

            total_time = (datetime.fromisoformat(events[-1]["timestamp"].rstrip('Z')).replace(tzinfo=timezone.utc) - first_event_time).total_seconds()
            logging.info(f"Replay completed. {total_events} events replayed.")
            logging.info(f"Original duration: {total_time:.2f}s, Replay duration: {total_time/replay_speed:.2f}s")
            
        except ValueError as e:
            logging.error(f"Error parsing timestamps: {str(e)}")
            logging.debug(f"Problematic timestamp format: {events[0]['timestamp']}")
            raise
            
    except FileNotFoundError:
        logging.error(f"Replay file not found: {replay_file}")
    except PermissionError:
        logging.error(f"Permission denied accessing file: {replay_file}")
    except Exception as e:
        logging.error(f"Unexpected error during replay: {str(e)}")

def test_openai_endpoint(openai_url: str, openai_model: str) -> bool:
    # Try to get a response from Ollama by listing running models
    # See: https://github.com/ollama/ollama/blob/main/docs/api.md#list-running-models
    # TODO: This is a temporary solution, as I'm not sure the endpoint is standard OpenAI compatible
    response = requests.get(f"{openai_url}/api/tags")
    if response.status_code == 200:
        # The endpoint is available. Now we need to check if the model is running
        # Implement model availability check and retry logic

        while True:
            logging.info(f"Endpoint is responding, checking if model {openai_model} is available...")
            # Response is a JSON object with a "models" field, which is a list of model objects
            if response.json().get("models") and any(model.get("name") == openai_model for model in response.json().get("models")):
                logging.info(f"Model {openai_model} is available")
                return True
            else:
                logging.info(f"Model {openai_model} not available, retrying in 30 seconds...")
                
                # Wait 30 seconds before retrying, but allow the user to interrupt
                try:
                    time.sleep(30)
                except KeyboardInterrupt:
                    logging.info("Interrupted by user, exiting...")
                    return False
    else:
        logging.error(f"Failed to get response from OpenAI endpoint: {response.status_code}")
        return False

def forward_to_openai(event: dict, openai_url: str, openai_model: str, model_seed: int, display_stats: bool = False, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
    # See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
    # According to Ollama's documentation, the response is expected to be a JSON object with the following fields:
    #
    # * total_duration: time spent generating the response
    # * load_duration: time spent in nanoseconds loading the model
    # * prompt_eval_count: number of tokens in the prompt
    # * prompt_eval_duration: time spent in nanoseconds evaluating the prompt
    # * eval_count: number of tokens in the response
    # * eval_duration: time in nanoseconds spent generating the response
    # * response: the generated response

    # Create an event message suitable for an LLM
    log_message = (
        f"Entity '{event['entity_id']}' changed state: "
        f"{event['old_state']} -> {event['new_state']}"
        f"{' (image data present)' if 'image_data' in event else ''}"
    )
    
    # Do we have image data?
    # Then the payload is a bit different. We call /api/generate with:
    # {
    #   "model": "llama3.2-vision",
    #   "prompt": "Summarise this picture. From a home security perspective, any suspicious activity",
    #   "stream": false,
    #   "options": {
    #     "seed": insert_model_seed_here
    #   },
    #   "images": ["insert_image_data_here"] 
    # }
 
    simplified_payload = {
        "model": openai_model,
        "options": {
            "seed": model_seed
        },
        "stream": False,
    }

    if "image_data" in event:
        # Quickly check if the image data is valid (check if we can decode it as base64)
        try:
            base64.b64decode(event["image_data"])
        except Exception as e:
            logging.error(f"Invalid image data, skipping: {str(e)}")
            return

        simplified_payload["images"] = [event["image_data"]]
        simplified_payload["prompt"] = "Summarise this picture. From a home security perspective, SPECIFICALLY provide any suspicious activity you find."
        logging.info(f"AI Request (image)")
    else:
        simplified_payload["system"] = system_prompt
        simplified_payload["prompt"] = log_message
        logging.info(f"AI Request: {log_message}")

    # Make the request and get the response.
    response = requests.post(
        openai_url + "/api/generate", 
        headers={"Content-Type": "application/json"},
        data=json.dumps(simplified_payload)
    )

    # Parse the response JSON
    if response.status_code == 200:
        result = response.json()
        # Extract just the assistant's message content
        result = result.get("response", {})
    else:
        logging.error(f"Failed to get response from OpenAI endpoint: {response.status_code}")
        # Write the request equivalent as a CURL command for the user to debug
        logging.info(f"CURL equivalent: curl -X POST '{openai_url}/api/generate' -H 'Content-Type: application/json' -d '{json.dumps(simplified_payload)}'")
        result = "{}"

    logging.info(f"AI Response: {result}")

    # TODO: Display stats, as they are only available in the non-simplified request
    # if display_stats:
    #    logging.info(f"Stats: {result.get('total_duration', 'N/A')}s, {result.get('load_duration', 'N/A')}ns, {result.get('prompt_eval_count', 'N/A')} tokens, {result.get('prompt_eval_duration', 'N/A')}ns, {result.get('eval_count', 'N/A')} tokens, {result.get('eval_duration', 'N/A')}ns")

def parse_args():
    parser = argparse.ArgumentParser(description="Utility to interact with Home Assistant.")
    
    # Global arguments that apply to all commands
    parser.add_argument(
        "--hass-url",
        type=str,
        help=f"URL of the Home Assistant instance (default: {DEFAULT_URL})",
        default=DEFAULT_URL
    )

    parser.add_argument(
        "--hass-access-token", 
        type=str,
        help=f"Access token for the Home Assistant instance",
        default=DEFAULT_TOKEN
    )

    # Subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='action',
        required=True,
        help='Available commands',
        title='commands'
    )

    # These arguments are shared by the listen and states commands
    hass_args = [
        ('--hass-url', {'type': str, 'help': f'URL of the Home Assistant instance (default: {DEFAULT_URL})', 'default': DEFAULT_URL}),
        ('--hass-access-token', {'type': str, 'help': 'Access token for the Home Assistant instance', 'default': DEFAULT_TOKEN})
    ]

    # Define commands and their arguments
    commands = {
        'listen': {
            'help': 'Listen for state changes',
            'args': hass_args + [
                ('--output', {'type': str, 'help': 'File to dump the output to (defaults to stdout)'}),
                ('--max-retries', {'type': int, 'default': DEFAULT_MAX_RETRIES, 'help': f'Maximum number of connection retry attempts (default: {DEFAULT_MAX_RETRIES})'}),
                ('--retry-delay', {'type': float, 'default': DEFAULT_RETRY_DELAY, 'help': f'Delay between retry attempts in seconds (default: {DEFAULT_RETRY_DELAY})'}),
                ('--append', {'type': bool, 'default': False, 'help': 'Append to the dump file instead of overwriting'})
            ]
        },
        'states': {
            'help': 'Get current states',
            'args': hass_args + [
                ('--output', {'type': str, 'help': 'File to dump the output to (defaults to stdout)'}),
                ('--timeout', {'type': int, 'default': DEFAULT_TIMEOUT, 'help': f'Timeout for requests in seconds (default: {DEFAULT_TIMEOUT})'})
            ]
        },
        'replay': {
            'help': 'Replay events from file, and optionally forward them to an OpenAI compatible Endpoint (e.g. Ollama)',
            'args': [
                ('--input', {'type': str, 'required': True, 'help': 'File to replay events from'}),
                ('--replay-speed', {'type': float, 'default': DEFAULT_REPLAY_SPEED, 'help': f'Speed multiplier for replay (default: {DEFAULT_REPLAY_SPEED})'}),
                ('--infer', {'type': bool, 'default': DEFAULT_INFER, 'help': 'Forward events to an OpenAI compatible Endpoint (e.g. Ollama)'}),
                ('--openai-url', {'type': str, 'default': DEFAULT_OPENAI_URL, 'help': f'URL of the OpenAI compatible (inference) endpoint (default: {DEFAULT_OPENAI_URL})'}),
                ('--openai-model', {'type': str, 'default': DEFAULT_OPENAI_MODEL, 'help': f'Model to use for inference (default: {DEFAULT_OPENAI_MODEL})'}),
                ('--model-seed', {'type': int, 'default': DEFAULT_MODEL_SEED, 'help': f'Seed to use for inference (default: {DEFAULT_MODEL_SEED})'}),
                ('--stats', {'type': bool, 'default': DEFAULT_DISPLAY_STATS, 'help': 'Display inference stats'}),
                ('--system-prompt', {'type': str, 'default': DEFAULT_SYSTEM_PROMPT, 'help': f'System prompt to use for inference (default is too long to be displayed)'})
            ]
        }
    }

    # Create subparsers for each command
    for cmd, config in commands.items():
        cmd_parser = subparsers.add_parser(cmd, help=config['help'])
        for arg_name, arg_config in config['args']:
            cmd_parser.add_argument(arg_name, **arg_config)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Command dispatch table
    commands = {
        'listen': lambda: asyncio.run(listen_to_home_assistant(
            url=args.hass_url,
            token=args.hass_access_token,
            dump_file=args.output,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            append=args.append
        )),
        'states': lambda: get_states(
            url=args.hass_url,
            token=args.hass_access_token,
            dump_file=args.output,
            timeout=args.timeout
        ),
        'replay': lambda: asyncio.run(replay_events(
            replay_file=args.input,
            replay_speed=args.replay_speed,
            openai_url=args.openai_url,
            openai_model=args.openai_model,
            model_seed=args.model_seed,
            display_stats=args.stats,
            system_prompt=args.system_prompt,
            infer=args.infer
        ))
    }

    # Execute selected command
    commands[args.action]()
