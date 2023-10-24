# ViGPTQA - State-of-the-Art LLMs for Vietnamese Question Answering: System Overview, Core Models Training, and Evaluations

The official repository for paper: ViGPTQA - State-of-the-Art LLMs for Vietnamese Question Answering: System Overview, Core Models Training, and Evaluations.

Accepted to EMNLP 2023 Industry Track.

To promote open research on large models within the Vietnamese NLP community, this project has open-sourced the Vietnamese large model with instruction fine-tuning. The project employs Vietnamese instructional data to fine-tune Vietnamese large language models, resulting in a significant enhancement of the model's ability to understand and execute instructions.

## Overview
> Large language models (LLMs) and their applications in low-resource languages (such as in Vietnamese) are limited due to lack of training data and benchmarking datasets. This paper introduces a practical real-world implementation of a question answering system for Vietnamese, called ViGPTQA, leveraging the power of LLM. Since there is no effective LLM in Vietnamese to date, we also propose, evaluate, and open-source an instruction-tuned LLM for Vietnamese, named ViGPT. ViGPT demonstrates exceptional performances, especially on real-world scenarios. We curate a new set of benchmark datasets that encompass both AI- and human-generated data, providing a comprehensive evaluation framework for Vietnamese LLMs. By achieving state-of-the-art results and approaching other multilingual LLMs, our instruction-tuned LLM underscores the need for dedicated Vietnamese-specific LLMs. Our open-source model supports customized and privacy-fulfilled Vietnamese language processing systems.

Join our benchmarking challenge: [DopikAI's LLM Challenge](https://aihub.vn/competitions/596?fbclid=IwAR21G61Kqm2t8_TdVjfuMN4fic-T41_tqS6OntQBMrdo3jHndEpNvFGzRhE#learn_the_details-evaluation).

## Requirements and Installation
#### Create virtual environment
```
python3.8 -m venv env_vigpt
```
	
#### Active virtual environment
```
source env_vigpt/bin/activate
```
#### Install packages
```
pip install -r requirements.txt
```
## ViGPTs Training

### Preparing data
* data/sample_law_text.jsonl. In this file, each line contains a JSON object {"prompt": law_text, "response": ""}, where the prompt value is the Vietnamese law text, and the response value is empty.
* data/sample_question_answering.jsonl. In this file, each line contains a JSON object {"prompt": question_text, "response": "answer_text"}, where the prompt value is the instruction/question for the chatbot, and the response value is the bot's answer to the corresponding question.

### Pre-training
To adapt the LLMs to a specific domain, we first pretrain the LLMs on monolingual data from that domain to enhance their knowledge of that domain. In this work, we acquired a pre-trained LLM with comprehensive knowledge of Vietnamese law by pretraining the LLM on 252,425 Vietnamese law-related documents.

```
python3 finetune.py --data_path data/sample_law_text.jsonl --base_model $BASE_MODEL_PATH --batch_size 128 --micro_batch_size 2 --val_set_size 200  --cutoff_len 1024 --num_epochs 1 --output_dir 'gptj6b-finetune-law'
```
### Checkpoint Export
After pretraining the LLM on specific domain data, we need to merge the LoRA weights back into the base model for exporting to Hugging Face format and subsequent fine-tuning. Please navigate to the "scripts" folder, modify certain path parameters, and execute the merge_lora_gptj.py script.
```
python3 scripts/merge_lora_gptj.py
```

### Supervised Fine-tuning
This step involves the direct application of PEFT to the GPT-J model, along with some code related to prompt construction and tokenization.
```
python3 finetune.py --data_path data/sample_question_answering.jsonl --base_model $BASE_MODEL_PATH --batch_size=128 --micro_batch_size 2 --cutoff_len 512 --num_epochs 1 --output_dir 'chat-gpt-j-6B'
```

## Inference
Use generation.py script to run inference in interactive mode. Please navigate to the "supervised_finetuning_model" folder, modify certain path parameters, and execute the generation.py script.
```
python3 generation.py
```

To benchmark the datasets mentioned in the paper, please navigate to the "evaluation" folder. We provide scripts for evaluating F1, BLEU-1, BLEU-4, and ROUGE-1 scores. Please go to [DopikAI's LLM Challenge](https://aihub.vn/competitions/596?fbclid=IwAR21G61Kqm2t8_TdVjfuMN4fic-T41_tqS6OntQBMrdo3jHndEpNvFGzRhE#learn_the_details-evaluation) for more details.

## Data Availability
Our dataset for training is available upon request for research purposes. Please send your information, including details about research usage and affiliations to EMAIL OF CORRESPONDENCE HERE.

## Citation
TBU

## Acknowledgments
The authors would like to thank the DopikAI Technology Company (http://DoPik.AI),  Quang-Anh Bui, Xuan-Vu Dinh, Tien-Tung Bui, Thanh-Tu Nguyen, and many annotators for their hard work to support the evaluation task. Without their support, the work would not have been possible.
