# Ask Manim

**Text-to-Mathematical Animation Tool**  
Generate **beautiful mathematical animations** from **natural language prompts** using **Groq’s LLaMA3 models** and render them with **Manim**.

## File Structure

| File                         | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `streamlit_main_360macky.py` | Main app file, handles UI and Groq API calls        |
| `utils.py`                   | Prompt formatting, Manim code extraction & cleanup  |
| `GenScene.py`                | Auto-generated Python file containing the animation |
| `.env`                       | Store your `GROQ_API_KEY` securely                  |
| `requirements.txt`           | Required dependencies                               |

## Sample Prompts

- `"Draw a red square"`
- `"Write Hello LLaMA inside a circle"`
- `"Write 'hello' in a square and 'world' in a circle side by side"`
- `"Display 'Hello Math!' in large white text and fade it out"`

## Run

```bash
pip install -r requirements.txt
```

> create `.env` file to store API-Key

```bash
#start the app
streamlit run streamlit_main_360macky.py
```

> Note: Manim may require system packages like ffmpeg, cairo, pango

- Generate Manim code
- Display .mp4 generated video it
- Let you download .py, .mp4 file for local rendering

### Models Used
- LLaMA3-70B
- LLaMA3-8B

### Videos
[Demo Video 1: Yellow Square Animate](./demo_video_1.mp4)
[Demo Video 2: Yellow Circle Animate](./demo_video_2.mp4)
[Demo Video 3: Text Animate](./demo_video_3.mp4)
<!-- https://github.com/user-attachments/assets/222a5ef1-63f6-41e6-82be-bf3ae6f643d7
https://github.com/user-attachments/assets/6befd4d8-98c8-485d-8f73-5fdb8bccc0c3 -->


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

