# Ask Almanack by Naval Ravikant

Ask questions to Almanack in natural language

Reference code https://github.com/hwchase17/notion-qa

Built with [LangChain](https://github.com/hwchase17/langchain)

# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

# What is in here?
- Sample (not all) data from https://www.navalmanack.com/almanack-of-naval-ravikant/table-of-contents 
- Python script to query Almanack with a question
- Code to deploy on StreamLit
- Instructions for ingesting your own dataset

## Example Data
This repo uses the [Almanack by Naval Ravikant](https://www.navalmanack.com/almanack-of-naval-ravikant/table-of-contents ) as an example.
Few pages were downloaded 22 March 2023 so may have changed slightly since then!

## Ask a question
In order to ask a question, run a command like:

```shell
python qa.py "What is judgment?"
```

You can switch out `What is Happiness` for any question of your liking!

This exposes a chat interface for interacting with a Almanack book.
IMO, this is a more natural and convenient interface for getting information.

## Code to deploy on StreamLit

The code to run the StreamLit app is in `main.py`. 
Note that when setting up your StreamLit app you should make sure to add `OPENAI_API_KEY` as a secret environment variable.

## Local changes - YHK
- Had to give full paths to docs.index, pkl files
- For 'Access Denied' error, need to add `python -m` before
- Had to add following lines in `qa.py` and `main.py`
```shell
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
```
- Command to run streamlit app was
```shell
python -m streamlit run <full path to main.py>
```

## Instructions for ingesting your own dataset

Save as txt some pages from Almanack site and put them chapter-wise in `almanack` directory
```

Run the following command to ingest the data.

```shell
python -m ingest
```

Boom! Now you're done, and you can ask it questions like:

```shell
python -m qa.py "What is judgement?"
```
