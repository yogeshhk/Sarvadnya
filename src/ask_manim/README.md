# Ask Manim

Text 2 Mathematics Animation tool

## 🛠 Core Development

Steps:
- Build UI^ with streamlit, it will have following components
	- Title, subtitle having mentions of 360macky and manim grant sanderson
	- Text box for prompt, 'Generate' button
	- Text area (not box) for displaying generated manim code (later can be made editable)
	- Video area to display it
- Use UI with ready LLM as is, see how code is displayed, how video is generated in the background and shown on UI
- Deploy to spaces for personal testing, it everything looks ok, go for fine-tuning
- fine-tuning:
	- Prepare data from jsonl to data-frame^
	- Use standard hugging face way of fine-tuning or Ludwig, run that model locally using LM studio
	
^Use prompt to generate the code, mention that prompt here as well

### 📦 Installation


To start the app, run:

```
streamlit run streamlit_main.py
```

## References
- [generative-manim Github repo 360macky](https://github.com/360macky/generative-manim), [Its app](https://generative-manim.streamlit.app/) not working though ![Good UI](./UI_360macky.png] [How it works](https://generative-manim.streamlit.app/%EF%B8%8F_How_it_works)
- [Star coder fine tuning](https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py) 
- [Manim Dataset from Hugging Face mediciresearch/manimation](https://huggingface.co/datasets/mediciresearch/manimation/raw/main/manimation_instruct_dataset.jsonl)
- [Star coder](https://github.com/bigcode-project/starcoder)
- [ManimGPT-AI-Powered Manim Assistance YesChat AI](https://www.yeschat.ai/gpts-ZxX3634E-ManimGPT) Ideal UI, shows video display on screen
- [fine-tuning-llama2-7b-code-generation-ludwig ](https://huggingface.co/Omid-sar/fine-tuning-llama2-7b-code-generation-ludwig)
- [Steps By Step Tutorial To Fine Tune LLAMA 2 With Custom Dataset Using LoRA And QLoRA Techniques](https://www.youtube.com/watch?v=Vg3dS-NLUT4) by Kris Naik
- [Efficient Fine-Tuning for Llama-v2-7b on a Single GPU](https://www.youtube.com/watch?v=g68qlo9Izf0) by Deep Learning AI
- [Manim UI by Rob Pruzan](https://github.com/RobPruzan/manim-ui)


## 🤲 Contributing

This is an open source project. If you want to be the author of a new feature, fix a bug or contribute with something new. Fork the repository and make changes as you like. Pull requests are warmly welcome.
