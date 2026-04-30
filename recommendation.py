import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

profiles = pd.read_csv("clean_resumes.csv").fillna("")
jobs = pd.read_csv("clean_job_postings.csv").fillna("")

# Avoids re-fitting the model or re-processing jobs
tfidf = joblib.load("tfidf_model.pkl")
job_vectors = joblib.load("job_vectors.pkl")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def get_recommendations(resume_idx, top_n=5):
    resume_text = profiles.iloc[resume_idx]['resume_text']
    q_vector = tfidf.transform([resume_text])
    
    scores = (q_vector @ job_vectors.T).toarray().flatten()
    sorted_indices = scores.argsort()[-100:][::-1]
    candidate_descrip = jobs.iloc[sorted_indices]['description'].tolist()

    # Semantic ranking
    resume_emb = sbert_model.encode(resume_text, convert_to_tensor=True)
    job_embs = sbert_model.encode(candidate_descrip, convert_to_tensor=True)

    # Calculate scores based on contextual meaning
    semantic_scores = util.cos_sim(resume_emb, job_embs).flatten()

    scored_candidates = []
    for i in range(len(sorted_indices)):
        scored_candidates.append({
            'idx': sorted_indices[i],
            'score': semantic_scores[i].item()
        })

    final_indices = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)[:top_n]
    
    results = []
    for i in final_indices:
        job_idx = i['idx']
        job_score = i['score']
        print(f"Score: {job_score:.4f} | {jobs.iloc[job_idx]['title']}")
        results.append({
            "job_id":       str(jobs.iloc[job_idx]['job_id']),
            "title":        jobs.iloc[job_idx]['title'],
            "company_name": jobs.iloc[job_idx]['company_name'],
            "description":  jobs.iloc[job_idx]['description'],
            "score":        job_score,
        })
    return results

get_recommendations(0,5)
