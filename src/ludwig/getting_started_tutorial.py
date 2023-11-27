# https://ludwig.ai/latest/getting_started/prepare_data/
# conda install cuda -c nvidia
# pip install git+https://github.com/Keith-Hon/bitsandbytes-windows
# pip install bitsandbytes --index-url=https://jllllll.github.io/bitsandbytes-windows-webui
# pip install bitsandbytes==0.40.2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
# conda install cudatoolkit
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv('./data/rotten_tomatoes.csv')
model = LudwigModel(config='rotten_tomatoes.yaml')
results = model.train(dataset=df)
print(results)