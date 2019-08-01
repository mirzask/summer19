# Part 1 - Create own dataset from Google Images

Show an image in `fastai` using `open_image(path)`.

## Methods for Downloading images

### From Lecture

Inspired by pyimagesearch.com.

Ctrl+Shift+J (Windows) or Cmd+Option+J (Mac) to open up javascript console when at desired Google Images search page.

Get list of all the URLs for each image using the following code in your JS console:

```javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```
Save the list of URLs into a file, e.g. `teddies.csv`.

### ImageDownloader widget

`ImageDownloader` from `fastai.widgets`. Documentation [here](https://docs.fast.ai/widgets.image_cleaner.html#ImageDownloader):

```python
path = Config.data_path()/'image_downloader'
os.makedirs(path, exist_ok=True)
ImageDownloader(path)
```

### download_google_images script

```python
# Setup path and labels to search for
path = Config.data_path()/'image_downloader'
labels = ['boston terrier', 'french bulldog']

# Download images
for label in labels: 
    download_google_images(path, label, size='>400*300', n_images=50)

# Build a databunch and train! 
src = (ImageList.from_folder(path)
       .split_by_rand_pct()
       .label_from_folder()
       .transform(get_transforms(), size=224))

db  = src.databunch(bs=16, num_workers=0)

learn = cnn_learner(db, models.resnet34, metrics=[accuracy])
learn.fit_one_cycle(3)
```




## General steps

1. Fit using `fit_one_cycle` with defaults
2. Save the model w/ `learn.save('model_name')`
3. Unfreeze the model
4. Use the learning rate finder - `learn.lr_find()`
5. Plot the learning rates - `learn.recorder.plot()`
6. Find strongest downward slope, which is consistently going down, pick about where it starts to drop as top LR. For top LR, general rule of thumb is to use `1e-4` or `3e-4`.

```python
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# Step 1
learn.fit_one_cycle(4)

# Step 2
learn.save('stage-1')

# Step 3
learn.unfreeze()

# Step 4
learn.lr_find()

# Step 5
learn.recorder.plot()

# Fit model w/ specified LRs
learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))

# Save
learn.save('stage-2')
```

## Cleaning Images

1. FastClass python package. See documentation [here]
2. `ImageCleaner` from `fastai.widgets`. [docs](https://docs.fast.ai/widgets.image_cleaner.html#ImageCleaner)
3. 

`ImageCleaner` example code:

```python
ds, idxs = DatasetFormatter().from_toplosses(learn)

# Load the widget
ImageCleaner(ds, idxs, path)

# `ImageCleaner` creates 'cleaned.csv', so need to create bunch from this
df = pd.read_csv(path/'cleaned.csv', header='infer')

# We create a databunch from our csv. We include the data in the training set and we don't use a validation set (DatasetFormatter uses only the training set)
np.random.seed(42)
db = (ImageList.from_df(df, path)
                   .split_none()
                   .label_from_df()
                   .databunch(bs=64))

# Fit model on "cleaned" images
learn = cnn_learner(db, models.resnet18, metrics=error_rate)
```

> Find duplicates using `duplicates = True` argument in `ImageCleaner`

Get and delete duplicates with `ImageCleaner`

```python
ds, idxs = DatasetFormatter().from_similars(learn, layer_ls=[0,7,1], pool=None)

# Load the widget to delete dupes
ImageCleaner(ds, idxs, path, duplicates=True)
```


# Production

If you're doing an image at a time, easy to just use a CPU (instead of GPU) for inference.

Switch to CPU:

```python
fastai.defaults.device = torch.device('cpu')
```

Create a predictor:

```python
learn.export()
```

`learn.export` creates a file named 'export.pkl' in the directory where we were working that contains everything we need to deploy our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).


# Part 2 - Common Errors + Gradient Descent from scratch

## Errors

Default learning rate in `fastai` is 0.003

| Problem                         | Clue                                                                   | Solution                                        |
| ------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------- |
| LR too high                     | :arrow_up: validation loss                                             | :arrow_down: LR; need to fit from scratch again |
| LR too low                      | _Slow_ improvement in `error_rate` +/- training loss > validation loss | :arrow_up: LR, e.g. :arrow_up by 10 or 100      |
| Training loss > Validation loss | Suggests :arrow_down: LR or :arrow_down: # of epochs                   | :arrow_up: LR or :arrow_up: # of epochs         |
| Too few epochs                  | Training loss > Validation loss                                        | :arrow_up: # of epochs                          |
| Too many epochs                 | Overfitting (error_rate improves, then worsens)                        |                                                 |

*Too low LR and Too few epochs can look the same, i.e. both give **Training loss > Validation loss**


## Stochastic Gradient Descent (from scratch)

