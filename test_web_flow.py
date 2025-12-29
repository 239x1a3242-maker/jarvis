#!/usr/bin/env python3
"""
Test web automation flow
"""

import asyncio
import yaml
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator

async def test_web_automation():
    # Load config
    with open('config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize orchestrator
    orchestrator = Orchestrator(config)
    await orchestrator.memory.initialize()
    await orchestrator.tool_registry.initialize()
    await orchestrator.llm.initialize()

    print("Testing web automation flow...")
    print("=" * 50)

    # Test web automation command
    test_input = "open browser google.com and search for dog images"
    print(f"Input: '{test_input}'")

    try:
        # Classify intent
        intent = await orchestrator.intent_classifier.classify(test_input)
        print(f"Intent classified: {intent}")

        # Create plan
        plan = await orchestrator.planner.create_plan(test_input, intent)
        print(f"Plan created: {plan}")

        # Execute plan (uncomment to actually run)
        print("Executing plan...")
        result = await orchestrator.execute_plan(plan)
        print(f"Execution result: {result}")

        # Generate response
        response = await orchestrator.generate_response(test_input, result, intent)
        print(f"Final response: {response}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_web_automation())