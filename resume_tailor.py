import json
import re
import torch
from transformers import pipeline
from recommendation import get_recommendations, profiles

llm = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=512,
)

#promt is pretty strict because otherwise model likes to add extra text
SYSTEM_PROMPT = """You are a resume coach. Given a resume and a job posting, return JSON only with these keys:
- keywords_to_add
- skills_to_highlight
- experience_framing
- summary_rewrite
- red_flags
No preamble. JSON only."""


def extract_json(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()

    # Grab the largest JSON-looking block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    return text[start:end + 1]


def safe_parse(json_str, job_title):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"[warn] JSON parse failed for {job_title}")
        print("snippet:", json_str[:150])
        return None


def fallback_feedback():
    # Ensures we never drop a job entirely
    return {
        "keywords_to_add": [],
        "skills_to_highlight": [],
        "experience_framing": "",
        "summary_rewrite": "",
        "red_flags": ["LLM output could not be parsed"],
    }


def get_feedback(resume_text, job):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"RESUME:\n{resume_text}\n\n"
            f"JOB TITLE: {job.get('title', 'N/A')}\n"
            f"COMPANY: {job.get('company_name', 'N/A')}\n"
            f"DESCRIPTION:\n{job.get('description', '')}\n\n"
            "Return JSON tailoring advice."
        )},
    ]

    output = llm(messages, temperature=0.3, do_sample=True, return_full_text=False)
    raw_text = output[0]["generated_text"]

    json_str = extract_json(raw_text)
    if json_str is None:
        print(f"[warn] no JSON found for {job.get('title')}")
        print("raw:", raw_text[:200])
        return fallback_feedback()

    parsed = safe_parse(json_str, job.get("title"))
    if parsed is None:
        return fallback_feedback()

    return parsed


def run_tailoring(resume_idx, top_n=5):
    resume_text = profiles.iloc[resume_idx]["resume_text"]
    jobs = get_recommendations(resume_idx, top_n)

    results = []
    for job in jobs:
        desc = job.get("description", "")
        if not desc.strip():
            print(f"[warn] missing description for {job.get('title')}")
            continue

        feedback = get_feedback(resume_text, job)

        results.append({
            "job_id": job.get("job_id"),
            "title": job.get("title"),
            "company": job.get("company_name"),
            "score": job.get("score"),
            **feedback,
        })

        print(f"{job.get('title')} ({job.get('score', 0):.3f}) processed")

    return results


if __name__ == "__main__":
    results = run_tailoring(0, 5)
    print(json.dumps(results, indent=2))
