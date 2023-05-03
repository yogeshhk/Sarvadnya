# YHK AMA (Ask-me-Anything)

This repo ([original](https://github.com/hwchase17/langchain-streamlit-template)) serves as a testbed to try out diffent LLMs, different fine-tuning data, etc.

## How to Run
`streamlit run streamlit_main.py`

From Pycharm: [Ref](https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide)
I found a way to at least run the code from the IDE (PyCharm in my case). The streamlit run code.py command can directly be called from your IDE. (The streamlit run code.py command actually calls python -m streamlit.cli run code.py, which was the former solution to run from the IDE.)

The -m streamlit run goes into the interpreter options field of the Run/Debug Configuration (this is supported by Streamlit, so has guarantees to not be broken in the future1), the code.py goes into the Script path field as expected. In past versions, it was also working to use -m streamlit.cli run in the interpreter options field of the Run/Debug Configuration, but this option might break in the future.

