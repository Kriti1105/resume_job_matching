#https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured
#https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
import kagglehub
import pandas as pd
import os
import nltk
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
 
nltk.download("stopwords")

resume_path = kagglehub.dataset_download("suriyaganesh/resume-dataset-structured")
jobs_path   = kagglehub.dataset_download("arshkon/linkedin-job-postings")

for root, dirs, files in os.walk(resume_path):
    for file in files:
        print(os.path.join(root, file))

for root, dirs, files in os.walk(jobs_path):
    for file in files:
        print(os.path.join(root, file))

people     = pd.read_csv(os.path.join(resume_path, "01_people.csv"))
abilities  = pd.read_csv(os.path.join(resume_path, "02_abilities.csv"))
education  = pd.read_csv(os.path.join(resume_path, "03_education.csv"))
experience = pd.read_csv(os.path.join(resume_path, "04_experience.csv"))
skills     = pd.read_csv(os.path.join(resume_path, "05_person_skills.csv"))

exp_grouped = (experience.dropna(subset=["title"]).groupby("person_id")["title"].apply(", ".join).reset_index())
exp_grouped = exp_grouped.rename(columns={"title": "experience_text"})
edu_grouped = (education.dropna(subset=["program"]).groupby("person_id")["program"].apply(", ".join).reset_index())
edu_grouped = edu_grouped.rename(columns={"program": "education_text"})
skills_grouped = (skills.dropna(subset=["skill"]).groupby("person_id")["skill"].apply(", ".join).reset_index())
skills_grouped = skills_grouped.rename(columns={"skill": "skills_text"})
abilities_grouped = (abilities.dropna(subset=["ability"]).groupby("person_id")["ability"].apply(", ".join).reset_index())
abilities_grouped = abilities_grouped.rename(columns={"ability": "abilities_text"})

profiles = people[["person_id"]].copy()
profiles = profiles.merge(exp_grouped, on="person_id", how="left")
profiles = profiles.merge(edu_grouped, on="person_id", how="left")
profiles = profiles.merge(skills_grouped, on="person_id", how="left")
profiles = profiles.merge(abilities_grouped, on="person_id", how="left")
print(profiles.head(2))

profiles["resume_text"] = ("Experience: " + profiles["experience_text"].fillna("") + ". " + "Education: " + profiles["education_text"].fillna("") + ". " + "Skills: " + profiles["skills_text"].fillna("") + ". " + "Abilities: " + profiles["abilities_text"].fillna(""))

punctuations = "\'\"\\,<>./?@#$%^&*_~/!()-[]{};:"
stop_words   = set(stopwords.words("english"))
tr_table     = str.maketrans("", "", punctuations)
 
for index, row in profiles.iterrows():
    line  = str(row["resume_text"])
    line  = line.translate(tr_table)           
    words = line.lower().strip().split()        
    words = [w for w in words if w not in stop_words]  
    profiles.loc[index, "resume_text"] = " ".join(words)
 
profiles = profiles.drop_duplicates(subset="resume_text").reset_index(drop=True)
 
print(len(profiles))

jobs = pd.read_csv(os.path.join(jobs_path, "postings.csv"))
jobs = jobs[["job_id", "title", "company_name", "description"]].copy()
jobs = jobs.dropna(subset=["description"])
 
for index, row in jobs.iterrows():
    line  = str(row["description"])
    line  = line.translate(tr_table)
    words = line.lower().strip().split()
    words = [w for w in words if w not in stop_words]
    jobs.loc[index, "description"] = " ".join(words)
 
jobs = jobs.drop_duplicates(subset="description").reset_index(drop=True)
 
print(len(jobs))

profiles.to_csv("clean_resumes.csv", index=False)
jobs.to_csv("clean_job_postings.csv", index=False)


# Recommends jobs by calculating the dot product of TF-IDF vectors
tfidf = TfidfVectorizer(max_features=10000)

all_text = pd.concat([profiles['resume_text'], jobs['description']])
tfidf.fit(all_text)

resume_vectors = tfidf.transform(profiles['resume_text'])
job_vectors = tfidf.transform(jobs['description'])

def get_recommendations(resume_idx, top_n=5):
    q = resume_vectors[resume_idx].toarray()
    d = job_vectors.toarray()
    
    scores = (q @ d.T).flatten()
    sorted_indices = reversed(scores.argsort()[-5:])
    
    for i in sorted_indices: 
        print(f"Score: {scores[i]:.4f} | {jobs.iloc[i]['title']}")

get_recommendations(0,5)

# Data Analysis and Pipeline
print("\nDataset Summary")
print(f"Number of resumes:       {len(profiles)}")
print(f"Number of job postings:  {len(jobs)}")

profiles["word_count"] = profiles["resume_text"].fillna("").apply(lambda x: len(x.split()))
jobs["word_count"]     = jobs["description"].fillna("").apply(lambda x: len(x.split()))

print(f"\nResume word count  – mean: {profiles['word_count'].mean():.0f}, "
      f"median: {profiles['word_count'].median():.0f}, "
      f"min: {profiles['word_count'].min()}, max: {profiles['word_count'].max()}")
print(f"Job desc word count – mean: {jobs['word_count'].mean():.0f}, "
      f"median: {jobs['word_count'].median():.0f}, "
      f"min: {jobs['word_count'].min()}, max: {jobs['word_count'].max()}")

print(f"\nTF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

# Top TF-IDF Terms 
feature_names = np.array(tfidf.get_feature_names_out())

# Mean TF-IDF weight across all resumes / all jobs
mean_resume = np.asarray(resume_vectors.mean(axis=0)).flatten()
mean_job    = np.asarray(job_vectors.mean(axis=0)).flatten()

top_n = 20
resume_top_idx = mean_resume.argsort()[-top_n:][::-1]
job_top_idx    = mean_job.argsort()[-top_n:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(range(top_n), mean_resume[resume_top_idx][::-1], color="#1a90bb")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(feature_names[resume_top_idx][::-1])
axes[0].set_xlabel("Mean TF-IDF Score")
axes[0].set_title("Top 20 Terms – Resumes")

axes[1].barh(range(top_n), mean_job[job_top_idx][::-1], color="#a00000")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(feature_names[job_top_idx][::-1])
axes[1].set_xlabel("Mean TF-IDF Score")
axes[1].set_title("Top 20 Terms – Job Postings")

plt.tight_layout()
plt.savefig("top_terms.png", dpi=150) # Top 20 TF-IDF terms per corpus
plt.close()

# Document length distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(profiles["word_count"], bins=40, color="#1a90bb", edgecolor="white")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Number of Resumes")
axes[0].set_title("Resume Length Distribution")

axes[1].hist(jobs["word_count"], bins=40, color="#a00000", edgecolor="white")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Number of Job Postings")
axes[1].set_title("Job Posting Length Distribution")

plt.tight_layout()
plt.savefig("doc_lengths.png", dpi=150)
plt.close()

# Pipeline output recommendation table
# Make DataFrame of the top-n recommended jobs for a resume
def get_recommendations_table(resume_idx, top_n=5):
    q = resume_vectors[resume_idx].toarray()
    d = job_vectors.toarray()
    scores = (q @ d.T).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]

    rows = []
    for rank, j in enumerate(top_idx, 1):
        rows.append({
            "Rank": rank,
            "Resume ID": profiles.iloc[resume_idx]["person_id"],
            "Resume Snippet": profiles.iloc[resume_idx]["resume_text"][:80] + "...",
            "Recommended Job": jobs.iloc[j]["title"],
            "Company": jobs.iloc[j].get("company_name", "N/A"),
            "Cosine Score": round(scores[j], 4),
        })
    return pd.DataFrame(rows)

# Pick a few diverse sample resumes (first, middle, last)
sample_indices = [0, len(profiles) // 4, len(profiles) // 2]
all_recs = pd.concat([get_recommendations_table(i) for i in sample_indices],
                      ignore_index=True)

print(all_recs.to_string(index=False))
all_recs.to_csv("recommendation_results.csv", index=False) # Top-5 job recommendations for sample resumes table