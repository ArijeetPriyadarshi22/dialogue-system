# Enhanced End-to-End Interactive XAI System (Offline Option with LLM)

import json
import random
import shap
import pandas as pd
import numpy as np
import os
import dice_ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dice_ml import Dice
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, \
    DataCollatorForSeq2Seq
from datasets import load_dataset
import csv

base_path = os.path.dirname(__file__)
file_path_titantic = os.path.join(base_path, "domain_knowledge", "titanic_knowledge.json")
file_path_credit = os.path.join(base_path, "domain_knowledge", "credit_knowledge.json")
file_path_diabetes = os.path.join(base_path, "domain_knowledge", "diabetes.json")
file_path_train_llm = os.path.join(base_path, "domain_knowledge", "train_llm.jsonl")

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

dataset = load_dataset("json", data_files=file_path_train_llm)["train"]


def preprocess(example):
    input_enc = tokenizer(example["input"], max_length=256, truncation=True, padding="max_length")
    output_enc = tokenizer(example["output"], max_length=64, truncation=True, padding="max_length")
    input_enc["labels"] = output_enc["input_ids"]
    return input_enc


tokenized_dataset = dataset.map(preprocess, remove_columns=["input", "output"])

training_args = TrainingArguments(
    output_dir="./finetuned-flan-t5-xai",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    save_strategy="epoch",
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
)

trainer.train()

# === Offline Model for Local Use (HuggingFace Pipeline) ===
# llm_offline = pipeline("text2text-generation", model="google/flan-t5-base")
fine_tuned_pipe = pipeline(
    "text2text-generation",
    model="./finetuned-flan-t5-xai/checkpoint-4",
    tokenizer=tokenizer
)

# === Load Domain Knowledge ===
with open(file_path_titantic) as f:
    titanic_knowledge = json.load(f)

with open(file_path_credit) as f:
    credit_knowledge = json.load(f)

with open(file_path_diabetes) as f:
    feature_descriptions = json.load(f)

gender = ["Male", "Female"]

# === Load and Prepare Titanic Dataset ===
titanic_data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
titanic_data = titanic_data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
titanic_data["Sex"] = LabelEncoder().fit_transform(titanic_data["Sex"])
X = titanic_data[["Pclass", "Sex", "Age", "Fare"]]
y = titanic_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)


# === Counterfactual Search ===
def generate_counterfactual(input_data):
    # Ensure DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T

    # Ensure input_data uses only training feature columns
    input_data = input_data[X_train.columns]

    # Define all continuous and categorical features correctly
    continuous = ["Pclass", "Sex", "Age", "Fare"]

    # Build DiCE Data object with correct feature metadata
    d = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=continuous,
        outcome_name='Survived'
    )

    m = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(d, m, method="random")

    query_instances = X_train[4:6]

    # Generate CF
    cf = exp.generate_counterfactuals(
        query_instances,
        total_CFs=3,
        desired_class="opposite",
        features_to_vary=["Sex"]
    )
    return cf


# === Domain Knowledge Integration ===
def retrieve_domain_knowledge(feature, domain="titanic"):
    source = titanic_knowledge if domain == "titanic" else credit_knowledge
    matches = [arg["text"] for arg in source["arguments"] if feature in arg["features"]]
    return matches if matches else "No domain knowledge available."


# === Prompt Generator ===
def build_prompt(dialogue_history, user_input, input_data, domain="titanic"):
    shap_vals = explainer.shap_values(pd.DataFrame([input_data]))
    pred_class = model.predict(pd.DataFrame([input_data]))[0]
    shap_vector = shap_vals[pred_class][0]
    top_idx = np.argmax(np.abs(shap_vector))
    top_feature = X.columns[top_idx]
    impact = shap_vector[top_idx]
    impact_desc = "increased" if impact > 0 else "decreased"

    cf = generate_counterfactual(input_data)
    cf_df = cf.cf_examples_list[0].final_cfs_df
    new_value = cf_df[top_feature].iloc[0]
    cf_input = cf_df.iloc[[0]].drop(columns=["Survived"], errors="ignore")
    new_pred = model.predict(cf_input)[0]
    cf_desc = f"Changing {top_feature} to {gender[new_value]} would have {'increased' if new_pred == 1 else 'decreased'} survival chances."

    knowledge_text = retrieve_domain_knowledge(top_feature.lower(), domain=domain)

    predictions = 'survive' if model.predict(pd.DataFrame([input_data]))[0] == 1 else 'not survive'

    return f"""
    You are an AI system explaining machine learning decisions in a conversation.
    Here are details about AI model and possible explanation 


    Input Features:
    - {input_data}

    Prediction: The model predicts this person would {predictions}.

    SHAP Explanation: The most influential feature was '{top_feature}', which {impact_desc} the outcome.
    Counterfactual Explanation: {cf_desc}
    Domain Knowledge: {knowledge_text}

    Dialogue so far:
    {dialogue_history}

    User: {user_input}
    Give explanation for user questions
    Be conversational and helpful. Explain clearly, but don’t overload with jargon unless the user asks for technical details.
    """


# === Offline LLM Response ===
def query_llm_offline(prompt):
    result = fine_tuned_pipe(prompt, max_length=300, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].split("System:")[-1].strip()


# === Dialogue System Entry Point ===
def run_dialogue():
    dialogue_history = ""
    candidates = []
    input_data = X_test.iloc[0].to_dict()
    print("--- Offline Interactive Titanic Explainer with Domain Knowledge ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        prompt = build_prompt(dialogue_history, user_input, input_data, domain="titanic")
        response = query_llm_offline(prompt)
        candidates.append(response)

        with open("output_csv", mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["candidate"])
            for cand in zip(candidates):
                writer.writerow([cand])
        print(f"AI: {response}")
        dialogue_history += f"\nUser: {user_input}\nSystem: {response}"


if __name__ == "__main__":
    run_dialogue()