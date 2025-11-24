# Enhanced End-to-End Interactive XAI System (Offline Option with LLM)

import json
import shap
import pandas as pd
import numpy as np
import os
import dice_ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dice_ml import Dice
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
from datetime import datetime

base_path = os.path.dirname(__file__)

# Create a timestamped folder and log file
log_base = os.path.join(base_path, "logs")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_base, f"session_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "dialogue_log.txt")

file_path_titantic = os.path.join(base_path, "domain_knowledge", "titanic_knowledge.json")
file_path_credit = os.path.join(base_path, "domain_knowledge", "credit_knowledge.json")
file_path_diabetes = os.path.join(base_path, "domain_knowledge", "diabetes.json")
file_path_train_llm = os.path.join(base_path, "domain_knowledge", "train_llm.jsonl")

# Load a better offline LLM
# llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#
# # Ensure you have enough RAM/GPU or use quantized versions (GGUF/GGML for llama.cpp etc.)
# model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
#
# llm_offline_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

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

#for logging user input and responses
def log_to_file(text):
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")

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

    prompt= f"""
    Instruction:
    You are an AI assistant explaining a prediction from a Titanic survival model..
 
    Input Features:
    - {input_data}

    Prediction: The model predicts this person would {predictions}.

    SHAP Explanation: The most influential feature was '{top_feature}', which {impact_desc} the outcome.
    Counterfactual Explanation: {cf_desc}
    Domain Knowledge: {knowledge_text}

    Dialogue so far:
    {dialogue_history}

    User: {user_input}
    Give a short explanation for user questions.Don’t overload with jargon unless the user asks for technical details.
    """
    return prompt.strip()


# === Offline LLM Response ===
def query_llm_offline(prompt):
   # response = llm_offline_pipe(prompt, max_length=512, do_sample=True, temperature=0.5, top_p=0.95)
    llm = Llama(model_path="/Users/arijeet/Downloads/llama-pro-8b.Q4_K_M.gguf")

    output = llm(
        f"### User: {prompt}\n### Assistant:",
        max_tokens=512,
        stop=["###", "User:"]
    )
    return output["choices"][0]["text"].strip()

#Evaluate the response
def evaluate_response_with_judge(prompt, response, user_input, judge_model_path="/Users/arijeet/Downloads/llama-pro-8b-instruct.Q3_K_L.gguf"):
    judge = Llama(model_path=judge_model_path)

    eval_prompt = f"""
        ### Instruction:
        You are an expert AI tasked with evaluating assistant responses.
        
        Rate the assistant's explanation based on:
        1. Clarity
        2. Correctness (logical and factual alignment with question)
        3. Helpfulness (how well it answers the user's intent)
        
        Provide a score and short justification.
        
        ### User Question:
        {user_input}
        
        ### Assistant's Response:
        {response}
        
        ### Evaluation Format:
        Rating: [Good | Acceptable | Needs Improvement]
        Reason: <Your short explanation here>
        
        ### Evaluation:
    """

    result = judge(eval_prompt, max_tokens=128, stop=["###", "\n\n"])
    return result["choices"][0]["text"].strip()


# === Dialogue System Entry Point ===
def run_dialogue():
    dialogue_history = ""
    input_data = X_test.iloc[0].to_dict()
    print("--- Offline Interactive Titanic Explainer with Domain Knowledge ---")
    log_to_file("=== NEW DIALOGUE SESSION STARTED ===")
    log_to_file(f"Timestamp: {timestamp}\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            log_to_file("=== SESSION ENDED ===\n")
            break
        prompt = build_prompt(dialogue_history, user_input, input_data, domain="titanic")
        response = query_llm_offline(prompt)
        print(f"AI: {response}")
        eval_feedback = evaluate_response_with_judge(prompt, response, user_input)
        print(f"\n[Judge Evaluation]: {eval_feedback}")

        # Log to file
        log_to_file(f"User: {user_input}")
        log_to_file(f"Prompt: {prompt}")
        log_to_file(f"AI Response: {response}")
        log_to_file(f"Evaluation: {eval_feedback}")
        log_to_file("-" * 50)

        dialogue_history += f"\nUser: {user_input}\nSystem: {response}"


if __name__ == "__main__":
    run_dialogue()