# Synthetic cover text generator

This repository provides a fine-tuning setup for **generative text steganography**.

Setup:
- Bring your own natural text dataset **D0** (e.g. WikiText-103)
- Fine-tune a LLM of your choice (e.g. Qwen3-8B) on D0 to match the statistical 
distribution of the text samples as accurately as possible
- Sample from the fine-tuned model to generate a synthetic cover text dataset **D1**,
which should look very similar to D0.
- Use the fine-tuned model along with a steganography method of your choice (e.g. 
arithmetic coding) to generate a stego text dataset **D2** based on D0.
- Use steganalysis methods of your choice to perform binary classification on the pairs 
D0-D2, D0-D1 and D1-D2, to evaluate the security of the steganography method and check it 
against a control group.


## How to use this repo

### 1. Virtual Environment
Before running any code:  
1. Install `uv` if you don't already have it on your device:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Install all specified packages:
```shell
uv sync
```

### 2. Add your own dataset (optional)
Make your dataset into a .txt file, in which each line is one text sample.
Then put the .txt file into the `data/` directory.

### 3. Fine-tune a LLM of your choice (optional)
⚠️ Depending on the size of the model and the dataset, this step may require one or more powerful GPUs.  
Choose the data file in `data/` and the huggingface model and run this command (adapt it as needed):
```shell
python3 -m src.train_lora --data_file your_data.txt --model_name your_huggingface_model
```

### 4. Generate a synthetic cover text dataset
Choose a fine-tuned model among those available in `fine_tuned_models/` and run this command (adapt it as needed):
```shell
python3 -m src.generate_synthetic_cover_text --model_path path_to_fine_tuned_model --num-samples 1000 --prompt ""
```
