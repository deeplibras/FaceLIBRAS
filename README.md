# FaceLIBRAS
FaceLIBRAS is a Convolutional Neural Network for facial expression classification.

# How to use
- Download this repository
- Import the facelibras() function inside the facelibras.py

```python
facelibras(database_path, nb_epoch, use_augmentation = None, resize = None)
```

- database_path: The path to database folder
- nb_epoch: The number of epochs to use
- use_augmentation: If should use image augmentation or not (defaul = False)
- resize: If the image need to be resize to 100x100 (default = True)

The function returns the train history and the trained model. You can use the functions in Utils.py to save the history as .CSV or plot

# Datasets
Read the [datasets creation](https://github.com/deeplibras/facelibras/tree/master/datasets)
