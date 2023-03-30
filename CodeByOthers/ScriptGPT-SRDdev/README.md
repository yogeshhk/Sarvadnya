# ScriptGPT
<a href="https://huggingface.co/SRDdev/Script_GPT">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97-Huggingface-yellow">
</a>
<a href="https://srddev-scriptify.hf.space">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow">
</a>

ScriptGPT is a GPT model built to generate amazing Youtube/Podcast/Film Scripts. ScriptGPT is a PyTorch implementation of the GPT (Generative Pre-trained Transformer) language model.


## 🧠 Introduction to GPT
GPT (Generative Pre-trained Transformer) is a language model developed by OpenAI. It is based on the transformer architecture, which was introduced in the paper "Attention is All You Need" by Google researchers. The key idea behind GPT is to pre-train a deep neural network on a large dataset, and then fine-tune it on a specific task, such as language translation or question answering.

GPT's architecture consists of an encoder and a decoder, both of which are made up of multiple layers of self-attention and feed-forward neural network. The encoder takes in the input sequence and produces a representation of it, while the decoder generates the output sequence based on the representation.

GPT-2, an updated version of GPT, was trained on a dataset of over 40 GB of text data, and is able to generate human-like text, complete tasks such as translation and summarization, and even create original content.

GPTLite is a smaller version which is built for fine-tuning and is trained on the Dataset, which is still powerful enough to generate human-like text, but with less computational resources required.

## 📂 Dataset
The model is currently in beta stage and is only trained on some youtube videos. As the dataset for this is not available, I am building the dataset myself.

__The model is still in beta phase and may generate illogical text.__


## ⚙️ Models
| Model Name         | Description                                        | Size   | Links                                                    |
|--------------------|----------------------------------------------------|--------|----------------------------------------------------------|
| Script_GPT         | Model for Generating Technical Content             | 100M   | [Link](https://huggingface.co/SRDdev/Script_GPT)         |
| ScriptGPT-small    | Generalized model for Video script generation      | 124M   | [Link](https://huggingface.co/SRDdev/ScriptGPT-small)    |
| ScriptGPT-medium   | Generalized model for Video script generation      | 437M   | [Link](https://huggingface.co/SRDdev/ScriptGPT-medium)   |



## 🖋️ Inference
__Load Model__
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SRDdev/Script_GPT")
model = AutoModelForCausalLM.from_pretrained("SRDdev/Script_GPT")
```

__Pipeline__
```python
from transformers import pipeline
generator = pipeline('text-generation', model= model , tokenizer=tokenizer)
```

__Call__
```python
context = "What is the meaning of life?"
length_to_generate = 200 

script = generator(context, max_length=length_to_generate, do_sample=True)[0]['generated_text']
```

## 🫂 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## 📝 Citations
```
@citation{ 
  Script_GPT,
  author = {Shreyas Dixit},
  year = {2023},
  url = {https://huggingface.co/SRDdev/Script_GPT}
}
```
