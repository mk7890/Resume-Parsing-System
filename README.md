# Resume Parsing System

This project implements a Natural Language Processing Transformer architecture-based model to extract key entities from a resume such as Skills, Experience, and Companies worked for, as well as scoring and ranking resumes based on a given job description. 

## Table of Contents:
1.	 Problem Statement
2.	Objective
3.	Features & capabilities
4.	Data Sourcing and Preparation 
5.	Modelling
6.	Evaluation

# Problem Statement (Overview)

In today’s competitive job market, recruiters face the daunting task of manually screening hundreds of resumes to identify the most qualified candidates for a given role. This process is not only time-consuming but also prone to human error, leading to missed opportunities and suboptimal hiring decisions. Additionally, the lack of standardized resume formats and the variability in how candidates present their skills and experiences further complicates the screening process. As a result, there is a critical need for an automated system that can efficiently parse resumes, extract key information, and rank candidates based on their relevance to specific job descriptions.

# Objective

The objective of this project is to develop an intelligent resume parsing and ranking system that leverages advanced natural language processing (NLP) techniques. Specifically, the system uses DistilBERT for Named Entity Recognition (NER) to extract critical information such as skills, experience, and education from resumes. It then employs TF-IDF vectorization and cosine similarity to evaluate and rank resumes based on their alignment with a given job description. The ultimate goal is to streamline the recruitment process by providing recruiters with a ranked list of candidates, enabling faster, more accurate, and data-driven hiring decisions.

# Features (Capabilities):

-	Entity recognition: Job role, Skills, etc
-	Resume Scoring based on job description
-	Batch resume ranking
-	Highlight missing skills in single resume analysis
-	
# Data:
Original dataset is a public jobs dataset available on Kaggle: https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
The dataset had over 1.6 million entries(rows) and 23 columns. Unnecessary columns were removed to retain only relevant data. The dataset used for the project consists of columns containing: 

Name : person name
Title : broader job category/classification
Role : job designation
Contact : phone number
Qualifications : education achievements such as P.hD, MBA
Experience : years of work experience
Skills : relevant skills
Company : company name

# Data Preprocessing Steps: 

1.	Data Cleaning.
-	Dropping irrelevant columns
-	Renaming remaining columns to meaningful labels
2.	Text preprocessing.
-	Lowercasing text for consistency
-	Remove special characters and extra whitespaces to improve model performance
3.	Reducing the dataset size to fewer entries by randomly sampling the data to accommodate for training resources.
4.	Creating Word-Level labels and tokens: these are BIO tags necessary for NER model training.
B – denotes the beginning of a named entity
I – denotes and intermediate/inside entitiy ( continuation of the named entity )
O – denotes and outside entity or no entity.
5.	Assigning Labels(numerical integers) and IDs(integers) to entity columns.
6.	Splitting the data into train, validation and test sets.
 	
# Modeling

A BERT (DistilBERT) model is used. BERT stands for Bidirectional Encoding Representations from Transformers. It’s a power model suitable for Natural Language Processing tasks because its able to understand context and semantics by learning both left to right and right to left.
Fine tuning is done by using the prepared dataset for custom Named Entity Recognition to learn to predict entities such as job roles and skills when fed text from a resume or job description.

# Model Evaluation

Evaluation is through F1 Score, Precision, Recall and Accuracy.
