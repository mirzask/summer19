Because I'm using this on the cluster, I will need to first activate the fastai virtual environment that I set up using the following code:

```
%%bash
source activate fastai
python -m ipykernel install --user --name fastai --display-name "Python (fastai)"
```

Magic commands to always include at the top of your Jupyter notebook

```
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

Multi-class classification, "fine grained classification" to classify the type of pet.

`untar_data` is a function to download and untar the data.

Use the `help(...)` function to see where function came from and more info. For example, `help(untar_data)`

## Python's pathlib

This functions similar to the `here` package in R. See [here](https://realpython.com/python-pathlib/) for more info.

Example:

```python
pathlib.Path.home() / 'Documents' / 'data' / 'something.py'

path = pathlib.Path('test.md')

path.parent
```

## Functions for images

- `get_image_files` - use this to get image file names
- `ImageDataBunch` gives labels + data. Contains a "bunch" of data, e.g. training, validation, +/- test
  - `ImageDataBunch.from_name_re` - labels are part of the file names, you can use RegEx
  - `ImageDataBunch.from_folder`
  - `ImageDataBunch.from_csv` - csv should contain 'name' (file path) and 'label', e.g. 0/1, columns. (Default name = labels.csv)
  - `ImageDataBunch.from_lists` - where labels are available in a list/array
  - Custom function using `ImageDataBunch.from_name_func` where you specify `label_func = lambda x: ...`
  - the `size = 224` (or some other number) needs to be specified b/c all images need to be of the same shape and size
  - Has a property called `c`, which can be thought of as the number of classes for classification problems. From the example below, it would be `data.c`
- `normalize` - data should be "normalized", i.e. same mean and std deviation
- Verify labels and images look ok using the `show_batch` method. For example, `data.show_batch(rows=3, figsize=(7,6))`


`ImageDataBunch` example:

```python
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img) # file names for each image
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
```

From the Lession 1 notes, below are the details of the arguments for the `ImageDataBunch` function:

- `path_img`: a path containing images
- `fnames`: a list of file names
- `pat`: a regular expression (i.e. pattern) to be used to extract the label from the file name
- `ds_tfm`: e.g. center/randomized cropping, padding and resizing all images to size 224
- `size`: what size images do you want to work with


## Fitting and Transfer Learning with CNN

### Create a learner:

```python
# Example 1
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# Example 2
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
```
> Always print out metrics on a validation set, which is created and available in the "data bunch" object.

### Fit the model

```python
learn.fit(1) # does 1 epoch

learn.fit_one_cycle(4) # does 4 epochs
```

**One cycle** learning is the new, best approach. [Paper](https://arxiv.org/abs/1803.09820)

### Save the model

```python
learn.save('my_model')
```

## Results

```python
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
```

Can visualize where the model misclassifies using `interp.plot_top_losses`:

```python
interp.plot_top_losses(9, figsize=(15,11))
```

Inspecting the confusion matrices and misclassifications

```python
#Plot confusion matrix
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# List "most confused"/misclassified labels
interp.most_confused(min_val=2)
```

## Unfreezing and Fine-tuning

Let's say you want to train some more after already fitting a model. We can _unfreeze_ the model and train some more.

```python
learn.unfreeze()

learn.fit_one_cycle(1)
```


Visualize the layers in a Conv Net - see paper by Zeiler and Fergus.

## Find the optimal learning rate

Use `learn.lr_find()`

```python
learn.load('stage-1')

learn.lr_find()

learn.recorder.plot()
```