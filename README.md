# cylearn

cylearn contains a collection of tools for machine learning. The 'cy' prefix is just the abbreviation of my given name.

## Installation

You can install cylearn through PyPI:

```
$ pip install cylearn
```

Or just copy the files you need to your project, but don't forget the LICENSE.

### Dependencies

+ [multiprocess](https://github.com/uqfoundation/multiprocess) (optional)

## Submodules

There is only one submodule so far.

+ `Data`
  + `Dataset`
  + `Loader`
## Data loading with `cylearn.Data`

`cylearn.Data` provides two classes `Dataset` and `Loader` and several functions `shuffle()`, `split()`, and `get_loader()`.

We first use some examples for a quick start before going into details. These examples are presented in the jupyter style.

### Example 1

```Python
import numpy as np
from cylearn.Data import Dataset, Loader
```

```Python
x = Dataset([1, 2, 3, 4, 5]).map(lambda i: i ** 2)
for _ in x: print(_)
```

```
1
4
9
16
25
```

```Python
loader_x = Loader(x, batch_size=2, shuffle=False)
for _ in loader_x: print(_)
```

```
[1, 4]
[9, 16]
[25]
```
