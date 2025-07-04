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

LLaMA3-70B
LLaMA3-8B

### Video

https://github.com/user-attachments/assets/222a5ef1-63f6-41e6-82be-bf3ae6f643d7
https://github.com/user-attachments/assets/6befd4d8-98c8-485d-8f73-5fdb8bccc0c3
