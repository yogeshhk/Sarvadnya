# FAQs Bot Generator
A custom FAQs Bot Generator, driven by `config.json`

```
{
  "APP_NAME": "MyApp",
  "DOCS_INDEX":"/fullpath/to/docs.index", 
  "FAISS_STORE_PKL":"/fullpath/to/faiss_store.pkl",
  "FILES_PATHS": [
    "/fullpath/to/file1.csv",
    "/fullpath/to/file2.txt",
    "/fullpath/to/file3.pdf"
  ]
}
```

Typically, both DOCS_INDEX and FAISS_STORE_PKL are stored in `./models` directory and data files are stored in `./data/` directory. Both directories need to be present.


You can manually edit it or run following wizard app to set the same parameters, it writes the same `config.json`.

```shell
python bot_config_wizard.py
```

FAQs Bot Generator reads the `config.json` and creates the FAQs Bot. To run the generator, use

```shell
streamlit run stremlit_main.py
```



## Prerequisites 

### General
- Create conda environment with python=3.10 (?) use `requirements.txt` for the setup
- Activating that environment, pip install google-cloud-aiplatform==1.27

### For running Vertex AI APIs
- Select or create a Google Cloud project.
- Make sure that billing is enabled for your project.
- Enable the Vertex AI API
- Create credentials json (Ref https://www.youtube.com/watch?v=rWcLDax-VmM)
- Set Environment variable GOOGLE_APPLICATION_CREDENTIALS as the above created json


