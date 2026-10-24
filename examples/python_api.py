"""Run after creating a detector: uv run python examples/python_api.py PATH."""
import json
import sys

from layerood.pipeline import OODPipeline

pipeline = OODPipeline.load(sys.argv[1], local_files_only=True)
print(json.dumps(pipeline.predict([
    "Can you tell me my account balance?",
    "How do I bake sourdough bread?",
]), indent=2))
