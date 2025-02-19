import asyncio
import json
import logging
import requests
from datetime import datetime, timezone
from retry import with_retries
from config import *

async def replay_events(replay_file: str | None = None, 
                        replay_speed: float = DEFAULT_REPLAY_SPEED, 
                        openai_url: str = DEFAULT_OPENAI_URL, 
                        openai_model: str = DEFAULT_OPENAI_MODEL, 
                        model_seed: int = DEFAULT_MODEL_SEED, 
                        display_stats: bool = DEFAULT_DISPLAY_STATS, 
                        system_prompt: str = DEFAULT_SYSTEM_PROMPT, 
                        infer: bool = DEFAULT_INFER) -> None:
    if not replay_file:
        logging.error("No replay file specified")
        return
    
    # Wish better types were a thing in Python
    # For future readers: yes, I like value-dependent types
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
                
                # Create human-readable output
                log_message = (
                    f"[{event['timestamp']}] ({idx}/{total_events}) "
                    f"Entity '{event['entity_id']}' changed state: "
                    f"{event['old_state']} -> {event['new_state']}"
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

def _check_api(url: str, model: str = None) -> bool:
    """Check API endpoint and optionally verify model availability."""
    try:
        response = requests.get(f"{url}/api/tags")
        response.raise_for_status()  # This will raise an exception for bad status codes
            
        if model:
            try:
                data = response.json()
                if not isinstance(data, dict) or "models" not in data:
                    logging.error(f"Unexpected API response format: {data}")
                    return False
                return any(m.get("name") == model for m in data["models"])
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from API: {response.text}")
                return False
        return True
    except requests.exceptions.RequestException as e:
        logging.debug(f"Request failed: {str(e)}")  # Changed to debug since retry will log the error
        raise  # Re-raise the exception for the retry mechanism
    except Exception as e:
        logging.debug(f"Unexpected error checking API: {str(e)}")  # Changed to debug
        raise  # Re-raise the exception for the retry mechanism

def test_openai_endpoint(openai_url: str, openai_model: str) -> bool:
    """Test OpenAI endpoint and model availability."""
    try:
        try:
            # First check if endpoint is responding
            with_retries(
                lambda: _check_api(openai_url),
                max_retries=5,
                delay=10,
                description="OpenAI endpoint connection"
            )()
        except Exception as e:
            logging.error("OpenAI endpoint is not responding after retries")
            return False
            
        logging.info("OpenAI endpoint is responding")
                
        try:
            # Then check if model is available
            with_retries(
                lambda: _check_api(openai_url, openai_model),
                max_retries=30,
                delay=30,
                description="Model availability check"
            )()
        except Exception as e:
            logging.error(f"Model {openai_model} is not available after retries")
            return False
            
        logging.info(f"Model {openai_model} is available")
        return True
        
    except Exception as e:
        logging.error(f"Failed to connect to OpenAI endpoint: {str(e)}")
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
    )
    
    # Format the payload
    simplified_payload = {
        "model": openai_model,
        "seed": model_seed,
        "stream": False,
        "system": system_prompt,
        "prompt": log_message
    }

    # Make the request and get the response. Time it.
    logging.info(f"AI Request: {log_message}")

    response = requests.post(
        openai_url + "/api/generate", 
        headers={"Content-Type": "application/json"},
        data=json.dumps(simplified_payload)
    )

    # Parse the response JSON
    if response.status_code == 200:
        result = response.json()
        logging.info(f"AI Response: {result.get('response', {})}")

        if display_stats:
            # Get timing metrics, defaulting to 'N/A' if not present
            metrics = {
                'total_duration': result.get('total_duration', 'N/A'),
                'load_duration': result.get('load_duration', 'N/A'),
                'prompt_eval_duration': result.get('prompt_eval_duration', 'N/A'),
                'eval_duration': result.get('eval_duration', 'N/A')
            }

            # Convert nanosecond timings to seconds
            metrics_s = {k: v/1e9 if isinstance(v, (int, float)) else 'N/A' 
                         for k, v in metrics.items()}

            # Get token counts
            prompt_eval_count = result.get('prompt_eval_count', 'N/A')
            eval_count = result.get('eval_count', 'N/A')
        
            logging.info(f"Stats: total_duration={metrics_s['total_duration']:.3f}s, load_duration={metrics_s['load_duration']:.3f}s, "
                        f"prompt_eval_count={prompt_eval_count} tokens, prompt_eval_duration={metrics_s['prompt_eval_duration']:.3f}s, "
                        f"eval_count={eval_count} tokens, eval_duration={metrics_s['eval_duration']:.3f}s")
    else:
        logging.error(f"Failed to get response from OpenAI endpoint: {response.status_code}")
        # Write the request equivalent as a CURL command for the user to debug
        logging.info(f"CURL equivalent: curl -X POST '{openai_url}/api/generate' -H 'Content-Type: application/json' -d '{json.dumps(simplified_payload)}'")
