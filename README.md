# Classify texts, detect texts by Jane Austen.

## Getting started
All code was tested on Python 3.9
To see the example of how this repository can be used, see `run_example.ipynb`
Scripts in this repositpry can be used via terminal as well. See the `args.py` for more details on script flags


## Repo's main scripts
- [ ]  `process.py` <- Performs basic EDA, preprocesses dataset and creates train / test subsets.
- [ ]  `train.py` <- trains and saves the model.
- [ ]  `args.py` <- arguments for all scripts, see this file to change the data preprocessing and / or training models parameners.


## Training Instructions
- [ ] Install requiremetns:
```
pip install -r requirements.txt
```
- [ ] Download NLTK lemmatizers
```
python 
>>> import nltk
>>> nltk.download('omw-1.4')
>>> nltk.download('wordnet')
```
- [ ] Run preprocessing:
```
python process.py 
```
python train.py --model_path <model-path>


- [ ] To see the TensorBoard visualizations of training use the following command:
```
tensorboard --logdir ./logs
```
This should generate visualizations on your localhost (if your model directory has no events, no visualization will be generated). Note that it will not work if you are using remote machine.

To see the saved outputs check the `run_example.ipynb` notebook (no TB visualization, you must run training for that)
