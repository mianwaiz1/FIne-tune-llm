Lung Cancer Question-Answering System
This project implements a question-answering system focused on lung cancer, utilizing a fine-tuned transformer model (google/flan-t5-small). It includes scripts for fine-tuning the model, evaluating its performance, and deploying an interactive chatbot using Streamlit. The dataset is provided in JSONL format (lung_cancer.jsonl).
Project Structure

finetune_llm.ipynb: Fine-tunes the google/flan-t5-small model on a custom lung cancer dataset.
finetune_evaluate.ipynb: Evaluates the base and fine-tuned models using ROUGE metrics.
frontend.py: Deploys an interactive chatbot interface for answering lung cancer-related questions.
lung_cancer.jsonl: Sample dataset containing question-answer pairs about lung cancer.

Requirements

Python 3.8+
Libraries:pip install torch transformers datasets evaluate streamlit


Hardware: GPU recommended for faster training (optional for inference).

Dataset
The dataset (lung_cancer.jsonl) is in JSONL format, where each line contains a JSON object with input (question) and output (answer) fields. Example:
{"input": "What is lung cancer?", "output": "Lung cancer is a type of cancer that begins in the lungs, often in the cells lining the air passages."}

Generating the Dataset

The provided lung_cancer.jsonl contains sample question-answer pairs.
To create your own dataset, ensure each line is a valid JSON object with input and output fields.

Scripts
1. Fine-Tuning (finetune_llm.ipynb)
This script fine-tunes the google/flan-t5-small model on the lung cancer dataset.
Steps:

Loads lung_cancer.jsonl into a Dataset object.
Tokenizes inputs and outputs (max length: 128 tokens).
Splits the dataset (90% train, 10% test).
Configures training arguments (e.g., learning rate, batch size).
Trains the model using the Trainer API.
Saves the fine-tuned model and tokenizer to ./fine_tuned_lung_cancer.
Includes a sample inference function (chat).

Usage:
jupyter or colab finetune_llm.ipynb

Output:

./fine_tuned_lung_cancer: Fine-tuned model and tokenizer.
./results: Training checkpoints.
./logs: Training logs.

Training Parameters:

Model: google/flan-t5-small
Learning Rate: 5e-5
Batch Size: 8
Epochs: 3
Weight Decay: 0.01
Evaluation/Save Steps: Every 500 steps

2. Evaluation (evaluate_models.py)
This script compares the performance of the base (google/flan-t5-small) and fine-tuned models using ROUGE metrics.
Steps:

Loads lung_cancer.jsonl and uses the training split for evaluation.
Generates answers for up to 50 samples using both models.
Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, etc.) for both models.
Prints evaluation results.

Usage:
jupyter or colab finetune_evaluate.ipynb

Notes:

Requires the fine-tuned model in ./fine_tuned_lung_cancer.
Uses training data for evaluation to ensure sufficient samples.

3. Chatbot Interface (frontend.py)
This script deploys an interactive chatbot using Streamlit to answer lung cancer-related questions.
Features:

Loads the fine-tuned model and tokenizer.
Provides a user-friendly chat interface.
Maintains chat history using Streamlit's session state.
Generates responses for user queries in real-time.

Usage:
streamlit run frontend.py

Notes:

Requires the fine-tuned model in ./fine_tuned_lung_cancer.
Run in a browser for the interactive UI.

Example Usage

Prepare the dataset: Place lung_cancer.jsonl in the project directory.
Fine-tune the model:jupyter or colab finetune_llm.ipynb

Evaluate models:jupyter or colab finetune_evaluate.ipynb


Run the chatbot:streamlit run frontend.py


Open the provided URL in a browser and ask questions like:
"What treatments are available for Stage 1 lung cancer?"
"What are the symptoms of lung cancer?"



Notes

Ensure the dataset is well-formatted and contains enough diverse question-answer pairs for effective fine-tuning.
Adjust hyperparameters in fine_tune_lung_cancer.py (e.g., learning_rate, num_train_epochs) based on dataset size and performance.
The chatbot is optimized for lung cancer-related questions but may not handle unrelated topics well.
For production, consider deploying the Streamlit app on a server or cloud platform.

Future Improvements

Expand the dataset with more diverse question-answer pairs.
Experiment with larger models (e.g., google/flan-t5-base) for better performance.
Add support for multi-turn conversations in the chatbot.
Incorporate additional evaluation metrics (e.g., BLEU, BERTScore).
