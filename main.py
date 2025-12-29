#!/usr/bin/env python3
"""
JARVIS - Personal AI Assistant
Main entry point for the JARVIS system.
"""

import asyncio
import logging
import yaml
from core.orchestrator import Orchestrator

def load_config():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config):
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )

async def main():
    config = load_config()
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting JARVIS...")

    orchestrator = Orchestrator(config)

    # Choose interface
    print("Welcome to JARVIS")
    print("Choose interface:")
    print("1. Text Interface")
    print("2. Voice Interface")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        if orchestrator.text_interface:
            await orchestrator.run_with_interface(orchestrator.text_interface)
        else:
            print("Text interface not enabled.")
    elif choice == '2':
        if orchestrator.voice_interface:
            await orchestrator.run_with_interface(orchestrator.voice_interface)
        else:
            print("Voice interface not enabled.")
    else:
        print("Invalid choice. Defaulting to text.")
        if orchestrator.text_interface:
            await orchestrator.run_with_interface(orchestrator.text_interface)

if __name__ == "__main__":
    asyncio.run(main())