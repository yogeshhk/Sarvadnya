# ChatBot Interface to Data-frame

- Accepts natural language queries like:
  - `What is the population of China?`
  - `Tell me the background of Algeria`
- Extracts **intent** and **entities** using **Rasa 3.x**
- Fetches answer from a CSV (`countries.csv`) using a **custom Rasa action**
- Supports both:
  - **Command Line** interaction (shell)
  - **Web UI** using **Flask + HTML/CSS/JS**
  -

## 🚀 Features

- Rasa 3.x (NLU + Core)
- Trained model with `nlu.yml`, `stories.yml`, `rules.yml`
- `actions.py` handles data queries
- Flask frontend (`app.py`) with chat interface
- REST API between Rasa backend and UI

## 🗂️ File Structure

| File / Folder         | Description                              |
| --------------------- | ---------------------------------------- |
| `app.py`              | Flask-based chat UI                      |
| `templates/home.html` | Frontend UI                              |
| `static/css/`         | Stylesheets                              |
| `static/js/bind.js`   | JS logic for chat                        |
| `data/countries.csv`  | Source country data                      |
| `actions/actions.py`  | Custom Rasa action                       |
| `nlu.yml`             | NLU training data                        |
| `rules.yml`           | Rule-based conversations                 |
| `stories.yml`         | Dialogue stories                         |
| `domain.yml`          | Rasa domain (intents, entities, actions) |
| `config.yml`          | Rasa pipeline & policy config            |
| `endpoints.yml`       | Rasa action server config                |
| `credentials.yml`     | Channel config (e.g., REST)              |

> `dfengine.py` is deprecated in Rasa 3.x setup — logic is migrated to `actions.py`

**Note**
Old Rasa .md/.json and new .yml format.
Web UI sends user messages to: POST http://localhost:5005/webhooks/rest/webhook

## 🛠️ Setup Instructions

### 1. Clone Repo & Setup

```bash
git clone <repo-url>
cd ask_dataframe
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Train Rasa Model

```bash
rasa train
```

### 3. Start Action Server (Terminal 1)

```bash
rasa run actions
```

### 4. Start Rasa Server (API) (Terminal 2)

```bash
rasa run --enable-api --cors "*" --debug
```

### 5. Start Flask Web UI (Terminal 3)

```bash
python app.py
Visit: http://localhost:8080/
```

Video (cmd based):
https://github.com/user-attachments/assets/b8d30d8f-f6c3-4ecb-8aac-b447f251faf3

Video (UI):
https://github.com/user-attachments/assets/bfa14b5e-d008-44ee-867b-7c059167937a

## References

- UI: Bhavani Ravi’s event-bot [code](https://github.com/bhavaniravi/rasa-site-bot), Youtube [Video](https://www.youtube.com/watch?v=ojuq0vBIA-g)
- Data gathering: Web Scraping with Python: Illustration with CIA World Factbook https://www.kdnuggets.com/2018/03/web-scraping-python-cia-world-factbook.html
