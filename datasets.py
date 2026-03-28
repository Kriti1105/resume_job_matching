#https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured
#https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
import kagglehub
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
 
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
