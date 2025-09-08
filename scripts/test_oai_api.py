import os
import sys
import time
import json

import tenacity
from openai import OpenAI

# Ensure we do not proxy the openai-proxy domain (helps inside corporate proxies)
os.environ.setdefault("NO_PROXY", "api.openai-proxy.org")

# -----------------------------------------------------------------------------
# OpenAI client
# -----------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai-proxy.org/v1"),
)


# @tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=2))
def call_moderation_api(text: str):
    """Call the moderation endpoint with automatic retries."""
    return client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )


def main():
    # ---------------------------------------------------------------------
    # Parse CLI arguments
    # ---------------------------------------------------------------------
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    else:
        sample_text = ''' i can't assist with that'''

    print("Input text:")
    print(sample_text)
    print("-" * 80)

    # ---------------------------------------------------------------------
    # Make request
    # ---------------------------------------------------------------------
    start_time = time.time()
    response = call_moderation_api(sample_text)
    elapsed_time = time.time() - start_time

    # ---------------------------------------------------------------------
    # Pretty-print results
    # ---------------------------------------------------------------------
    try:
        # openai>=1.0 uses Pydantic models which expose `model_dump`.
        raw_response = response.model_dump()
    except AttributeError:
        # Fallback for older versions – convert via json then back to dict.
        raw_response = json.loads(response.json()) if hasattr(response, "json") else str(response)

    print("Raw moderation response:")
    print(json.dumps(raw_response, indent=2, ensure_ascii=False))
    print("-" * 80)

    # Extract category scores
    raw_scores = response.results[0].category_scores
    if hasattr(raw_scores, "model_dump"):
        category_scores = raw_scores.model_dump()
    elif hasattr(raw_scores, "dict"):
        category_scores = raw_scores.dict()
    else:
        # Fallback: try to coerce to dict directly
        try:
            category_scores = dict(raw_scores)
        except Exception:
            category_scores = {k: v for k, v in raw_scores.items()}

    total_score = sum(category_scores.values())

    print("Category scores:")
    for category, score in category_scores.items():
        print(f"{category:25s}: {score:.4f}")

    print(f"\nTotal aggregated score: {total_score:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    main()
