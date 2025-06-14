
# 🤖 Motivational Chatbot Fine-Tuning using LoRA and PEFT

This project fine-tunes the [Falcon-RW-1B](https://huggingface.co/tiiuae/falcon-rw-1b) language model to generate motivational messages using Low-Rank Adaptation (LoRA) with the [PEFT](https://github.com/huggingface/peft) library.

---

## 📂 Files Included

- `motivation_data.jsonl` – A curated dataset of motivational quotes in JSON Lines format
- `finetune_motivational_bot.py` – Python script for LoRA fine-tuning using Hugging Face’s `transformers`, `peft`, and `datasets`

---

## 🚀 Technologies Used

- Python 3.10+
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- Accelerate
- BitsAndBytes
- Falcon-RW-1B

---

## 🔧 How to Run

> ⚠️ Requires a GPU-enabled environment (e.g., Google Colab, Kaggle, or local machine with CUDA)

```bash
# Clone the repo
git clone https://github.com/your-username/motivationalbot.git
cd motivationalbot

# Install dependencies
pip install -q transformers peft accelerate datasets bitsandbytes

# Run training script
python finetune_motivational_bot.py

```
## 💡 Dataset Preview

```bash
{"text": "Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle."}

{"text": "Success is not final, failure is not fatal: It is the courage to continue that counts."
```

✅ Goals

----
Learn LoRA-based fine-tuning

Apply PEFT to a real-world micro-task

Create a lightweight motivational chatbot model


