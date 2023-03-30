# HolyBot

🚀 HolyBot gives answers and relevant verses for your queries based on Bhagwad Gita/ Quran/ Bible.

This project is built using Pinecone and OpenAI ChatGPT. 
- Get your pinecone api key, environment [here](https://app.pinecone.io/organizations/-NMeB1SFhOs6iUwHoAtz/projects/us-east1-gcp:e749001/indexes)
- Get your OpenAI API key, organisation [here](https://platform.openai.com/account/api-keys)

## Usage

### Step - 1:
```
pip install requirements.txt
```

### Step - 2:

Get necessary API Keys.

```
PINECONE_API_KEY = "YOUR-PINECONE-API-KEY"
PINECONE_ENVIRONMENT = "YOUR-PINECONE-ENVIRONMENT"

OPENAI_API_KEY = "YOUR-OPENAI-API-KEY"
OPENAI_ORGANIZATION = "YOUR-OPENAI-ORGANIZATION"

HOLY_BOOK = "gita" ('bible'/ 'quran')
```

### Step - 3:

Create index for the selected holybook (gita/ bible/ quran).

```
python createindex.py --holybook $HOLY_BOOK --pinecone_apikey $PINECONE_API_KEY --pinecone_environment $PINECONE_ENVIRONMENT --openaikey $OPENAI_API_KEY --openaiorg $OPENAI_ORGANIZATION
```

### Step - 4:

Launch Gradio app.

```
python app.py --holybook $HOLY_BOOK --pinecone_apikey $PINECONE_API_KEY --pinecone_environment $PINECONE_ENVIRONMENT --openaikey $OPENAI_API_KEY
```
## Demo:

https://user-images.githubusercontent.com/12198101/224573190-7c10fad3-ca8b-4df9-8e3f-c36566dfc0d0.mov

## HuggingFace Space:

You can check out HolyBot on huggingface spaces - https://huggingface.co/spaces/ravithejads/HolyBot
