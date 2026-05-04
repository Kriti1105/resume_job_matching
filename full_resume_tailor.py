import argparse
import json
import os
import re
from pathlib import Path

import joblib
import kagglehub
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# File paths
CLEAN_RESUMES_PATH = Path("clean_resumes.csv")
CLEAN_JOBS_PATH = Path("clean_job_postings.csv")
TFIDF_PATH = Path("tfidf_model.pkl")
JOB_VECTORS_PATH = Path("job_vectors.pkl")
OUTPUT_JSON_PATH = Path("tailoring_results.json")

# Preprocessing
def clean_text(text, stop_words, tr_table):
    line = str(text).translate(tr_table)
    words = line.lower().strip().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def preprocess_data(force=False):
    required_outputs = [
        CLEAN_RESUMES_PATH,
        CLEAN_JOBS_PATH,
        TFIDF_PATH,
        JOB_VECTORS_PATH,
    ]

    if not force and all(path.exists() for path in required_outputs):
        print("[info] Preprocessed files already exist. Skipping preprocessing.")
        return

    print("[info] Running preprocessing...")

    nltk.download("stopwords")

    # Datasets
    resume_path = kagglehub.dataset_download("suriyaganesh/resume-dataset-structured")
    jobs_path = kagglehub.dataset_download("arshkon/linkedin-job-postings")

    people = pd.read_csv(os.path.join(resume_path, "01_people.csv"))
    abilities = pd.read_csv(os.path.join(resume_path, "02_abilities.csv"))
    education = pd.read_csv(os.path.join(resume_path, "03_education.csv"))
    experience = pd.read_csv(os.path.join(resume_path, "04_experience.csv"))
    skills = pd.read_csv(os.path.join(resume_path, "05_person_skills.csv"))

    exp_grouped = (
        experience.dropna(subset=["title"])
        .groupby("person_id")["title"]
        .apply(", ".join)
        .reset_index()
        .rename(columns={"title": "experience_text"})
    )

    edu_grouped = (
        education.dropna(subset=["program"])
        .groupby("person_id")["program"]
        .apply(", ".join)
        .reset_index()
        .rename(columns={"program": "education_text"})
    )

    skills_grouped = (
        skills.dropna(subset=["skill"])
        .groupby("person_id")["skill"]
        .apply(", ".join)
        .reset_index()
        .rename(columns={"skill": "skills_text"})
    )

    abilities_grouped = (
        abilities.dropna(subset=["ability"])
        .groupby("person_id")["ability"]
        .apply(", ".join)
        .reset_index()
        .rename(columns={"ability": "abilities_text"})
    )

    profiles = people[["person_id"]].copy()
    profiles = profiles.merge(exp_grouped, on="person_id", how="left")
    profiles = profiles.merge(edu_grouped, on="person_id", how="left")
    profiles = profiles.merge(skills_grouped, on="person_id", how="left")
    profiles = profiles.merge(abilities_grouped, on="person_id", how="left")

    profiles["resume_text"] = (
        "Experience: " + profiles["experience_text"].fillna("") + ". "
        "Education: " + profiles["education_text"].fillna("") + ". "
        "Skills: " + profiles["skills_text"].fillna("") + ". "
        "Abilities: " + profiles["abilities_text"].fillna("")
    )

    punctuations = "\'\"\\,<>./?@#$%^&*_~/!()-[]{};:"
    stop_words = set(stopwords.words("english"))
    tr_table = str.maketrans("", "", punctuations)

    profiles["resume_text"] = profiles["resume_text"].apply(
        lambda text: clean_text(text, stop_words, tr_table)
    )
    profiles = profiles.drop_duplicates(subset="resume_text").reset_index(drop=True)

    jobs = pd.read_csv(os.path.join(jobs_path, "postings.csv"))
    jobs = jobs[["job_id", "title", "company_name", "description"]].copy()
    jobs = jobs.dropna(subset=["description"])

    jobs["description"] = jobs["description"].apply(
        lambda text: clean_text(text, stop_words, tr_table)
    )
    jobs = jobs.drop_duplicates(subset="description").reset_index(drop=True)

    tfidf = TfidfVectorizer(max_features=10000)
    all_text = pd.concat([profiles["resume_text"], jobs["description"]])
    tfidf.fit(all_text)

    job_vectors = tfidf.transform(jobs["description"])

    profiles.to_csv(CLEAN_RESUMES_PATH, index=False)
    jobs.to_csv(CLEAN_JOBS_PATH, index=False)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(job_vectors, JOB_VECTORS_PATH)

    print("[info] Preprocessing finished.")
    print(f"[info] Resumes: {len(profiles)}")
    print(f"[info] Job postings: {len(jobs)}")

# Recommendation
def load_recommendation_assets():
    profiles = pd.read_csv(CLEAN_RESUMES_PATH).fillna("")
    jobs = pd.read_csv(CLEAN_JOBS_PATH).fillna("")
    tfidf = joblib.load(TFIDF_PATH)
    job_vectors = joblib.load(JOB_VECTORS_PATH)
    return profiles, jobs, tfidf, job_vectors


def get_recommendations(resume_idx, top_n, profiles, jobs, tfidf, job_vectors, sbert_model):
    resume_text = profiles.iloc[resume_idx]["resume_text"]
    q_vector = tfidf.transform([resume_text])

    # TF-IDF filtering
    scores = (q_vector @ job_vectors.T).toarray().flatten()
    sorted_indices = scores.argsort()[-100:][::-1]
    candidate_descriptions = jobs.iloc[sorted_indices]["description"].tolist()

    # semantic ranking with SentenceTransformer
    resume_emb = sbert_model.encode(resume_text, convert_to_tensor=True)
    job_embs = sbert_model.encode(candidate_descriptions, convert_to_tensor=True)
    semantic_scores = util.cos_sim(resume_emb, job_embs).flatten()

    scored_candidates = []
    for i, job_idx in enumerate(sorted_indices):
        scored_candidates.append({
            "idx": job_idx,
            "score": semantic_scores[i].item(),
        })

    final_candidates = sorted(
        scored_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )[:top_n]

    results = []
    for item in final_candidates:
        job_idx = item["idx"]
        job_score = item["score"]

        print(f"Score: {job_score:.4f} | {jobs.iloc[job_idx]['title']}")

        results.append({
            "job_id": str(jobs.iloc[job_idx]["job_id"]),
            "title": jobs.iloc[job_idx]["title"],
            "company_name": jobs.iloc[job_idx]["company_name"],
            "description": jobs.iloc[job_idx]["description"],
            "score": job_score,
        })

    return results

# Resume tailoring with LLM
SYSTEM_PROMPT = """You are a resume coach. Given a resume and a job posting, return JSON only with these keys:
- keywords_to_add
- skills_to_highlight
- experience_framing
- summary_rewrite
- red_flags
No preamble. JSON only."""


def extract_json(text):
    text = re.sub(r"```(?:json)?|```", "", text).strip()

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
    return {
        "keywords_to_add": [],
        "skills_to_highlight": [],
        "experience_framing": "",
        "summary_rewrite": "",
        "red_flags": ["LLM output could not be parsed"],
    }


def get_feedback(resume_text, job, llm):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"RESUME:\n{resume_text}\n\n"
                f"JOB TITLE: {job.get('title', 'N/A')}\n"
                f"COMPANY: {job.get('company_name', 'N/A')}\n"
                f"DESCRIPTION:\n{job.get('description', '')}\n\n"
                "Return JSON tailoring advice."
            ),
        },
    ]

    output = llm(
        messages,
        temperature=0.3,
        do_sample=True,
        return_full_text=False,
        max_new_tokens=512,
    )

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


def run_tailoring(resume_idx, top_n, profiles, jobs, tfidf, job_vectors):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[info] Loading SentenceTransformer model...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    recommended_jobs = get_recommendations(
        resume_idx=resume_idx,
        top_n=top_n,
        profiles=profiles,
        jobs=jobs,
        tfidf=tfidf,
        job_vectors=job_vectors,
        sbert_model=sbert_model,
    )

    print("[info] Loading Qwen LLM...")
    llm = pipeline(
        task="text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    resume_text = profiles.iloc[resume_idx]["resume_text"]

    results = []
    for job in recommended_jobs:
        desc = job.get("description", "")

        if not desc.strip():
            print(f"[warn] missing description for {job.get('title')}")
            continue

        feedback = get_feedback(resume_text, job, llm)

        results.append({
            "job_id": job.get("job_id"),
            "title": job.get("title"),
            "company": job.get("company_name"),
            "score": job.get("score"),
            **feedback,
        })

        print(f"{job.get('title')} ({job.get('score', 0):.3f}) processed")

    return results

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Run the full resume-job matching and tailoring pipeline."
    )
    parser.add_argument("--resume_idx", type=int, default=0)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Re-download/reprocess data even if saved files already exist.",
    )
    args = parser.parse_args()

    preprocess_data(force=args.force_preprocess)

    profiles, jobs, tfidf, job_vectors = load_recommendation_assets()

    if args.resume_idx < 0 or args.resume_idx >= len(profiles):
        raise IndexError(
            f"resume_idx must be between 0 and {len(profiles) - 1}, "
            f"but got {args.resume_idx}"
        )

    results = run_tailoring(
        resume_idx=args.resume_idx,
        top_n=args.top_n,
        profiles=profiles,
        jobs=jobs,
        tfidf=tfidf,
        job_vectors=job_vectors,
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFinal tailoring results:")
    print(json.dumps(results, indent=2))
    print(f"\n[info] Saved results to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
