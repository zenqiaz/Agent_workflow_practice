"""
Quick test: verify LLM connectivity for both dev and local configs.
Run from the working directory.

Usage:
    python test_llm_configs.py            # tests current ENV_FILE (or .env)
    python test_llm_configs.py --both     # tests both .env and .env.local
"""

import os
import sys
import json
from dotenv import dotenv_values

def test_config(env_file: str) -> bool:
    cfg = dotenv_values(env_file)
    model = cfg.get("LLM_MODEL", "gpt-4.1-mini")
    base_url = cfg.get("LLM_BASE_URL", "").strip() or None
    api_key = cfg.get("OPENAI_API_KEY", "missing")

    print(f"\n{'='*50}")
    print(f"  Config : {env_file}")
    print(f"  Model  : {model}")
    print(f"  Base   : {base_url or '(OpenAI default)'}")
    print(f"{'='*50}")

    try:
        from openai import OpenAI
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            max_tokens=10,
            temperature=0,
        )
        reply = (resp.choices[0].message.content or "").strip()
        print(f"  Reply  : {reply!r}")
        ok = "ok" in reply.lower() or len(reply) > 0
        print(f"  Status : {'PASS' if ok else 'FAIL (unexpected reply)'}")
        return ok
    except Exception as e:
        print(f"  Status : FAIL — {e}")
        return False


if __name__ == "__main__":
    test_both = "--both" in sys.argv

    if test_both:
        results = {}
        for f in [".env", ".env.local"]:
            if os.path.exists(f):
                results[f] = test_config(f)
            else:
                print(f"\n[skip] {f} not found")
        print("\n--- Summary ---")
        for f, ok in results.items():
            print(f"  {f:12s}  {'PASS' if ok else 'FAIL'}")
    else:
        env_file = os.environ.get("ENV_FILE", ".env")
        test_config(env_file)
