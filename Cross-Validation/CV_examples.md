# K-Fold CV

```bash
python kfold.py -i penguins.csv
```

# Stratified K-Fold CV

```bash
python stratified_kfold.py -i penguins.csv --label_column_name species
```

Confirm that folds are stratified by 'species'

```python
import pandas as pd

df = pd.read_csv('penguins_folds.csv')

df.groupby(['species', 'kfold']).size()
```