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