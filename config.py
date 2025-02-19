#!/usr/bin/env python3

import os

# System prompt used for security risk assessment
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

# Configuration constants with environment fallbacks
DEFAULT_URL = os.getenv("HAMOCK_HASS_URL", "localhost:8123")
DEFAULT_TOKEN = os.getenv("HAMOCK_HASS_ACCESS_TOKEN", "eyJhbGci...")
DEFAULT_TIMEOUT = int(os.getenv("HAMOCK_TIMEOUT", "10"))
DEFAULT_RETRY_DELAY = int(os.getenv("HAMOCK_RETRY_DELAY", "5"))
DEFAULT_MAX_RETRIES = int(os.getenv("HAMOCK_MAX_RETRIES", "3"))
DEFAULT_REPLAY_SPEED = float(os.getenv("HAMOCK_REPLAY_SPEED", "1.0"))
DEFAULT_OPENAI_URL = os.getenv("HAMOCK_OPENAI_URL", "http://localhost:11434")
DEFAULT_OPENAI_MODEL = os.getenv("HAMOCK_OPENAI_MODEL", "llama3.2:1b-instruct-q4_K_M")
DEFAULT_MODEL_SEED = int(os.getenv("HAMOCK_MODEL_SEED", "36424"))
DEFAULT_SYSTEM_PROMPT = os.getenv("HAMOCK_SYSTEM_PROMPT", SYSTEM_PROMPT)
DEFAULT_INFER = bool(os.getenv("HAMOCK_INFER", "false").lower() == "true")
DEFAULT_DISPLAY_STATS = bool(os.getenv("HAMOCK_DISPLAY_STATS", "false").lower() == "true") 