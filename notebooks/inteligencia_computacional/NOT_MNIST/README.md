## DEPENDÊNCIAS
- Python version 3.5
- `numpy` version 1.10 or later: http://www.numpy.org/
- `scipy` version 0.16 or later: http://www.scipy.org/
- `matplotlib` version 1.4 or later: http://matplotlib.org/
- `pandas` version 0.16 or later: http://pandas.pydata.org
- `scikit-learn` version 0.15 or later: http://scikit-learn.org
- `keras` version 2.0 or later: http://keras.io
- `tensorflow` version 1.0 or later: https://www.tensorflow.org
- `ipython`/`jupyter` version 4.0 or later, with notebook support


## CRIAR AMBIENTE
conda env create -f deep-learning.yml

## INSTRUÇÕES
Para baixar e extrair o dataset, utilize o comando:

```bash
python set_notmnist_pickle.py
```

Este processo pode demorar algum tempo. Como resultado, o arquivo notMNIST.pickle será gerado e refere-se a serialização dos valores de treino, validação e teste e que poderão ser lidos(utilizando o util/notMNIST.py)  

```python
notmnist_path = "notMNIST.pickle"  
(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = notMNIST.load_data(notmnist_path)
```
