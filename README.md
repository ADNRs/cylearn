# cylearn

cylearn contains a collection of tools for machine learning. The 'cy' prefix is just the abbreviation of my given name.

## Warning

This package is still under development, use with caution.

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
x = Dataset(np.arange(5)).map(lambda i: i ** 2)
for _ in x: print(_)
```

```
0
1
4
9
16
```

```Python
loader_x = Loader(x, batch_size=2, shuffle=False)
for _ in loader_x: print(_)
```

```
[0, 1]
[4, 9]
[16]
```

### Example 2

```
+- data/
|  +- 1.png
|  +- 1.txt
|  +- 2.png
|  +- 2.txt
|  |  ...
|  +- 5.png
|  +- 5.txt
|
+- demo.py
```

The above is the structure of an imaginary folder, where x.png is an image and x.txt stores an integer which is the label of x.png.

Assume the memory can only store two images once at a time because these images are too large. We now demonstrate how to make batches for these images and labels with `Dataset` and `Loader`.

```Python
import os
import numpy
from cylearn.Data import Dataset, get_loader
```

```Python
# Filter the correct files by their extension and recover their dirname.
images = Dataset(os.listdir('./data/')).filter(lambda f: '.png' in f).map(lambda f: './data/' + f)
labels = Dataset(os.listdir('./data/')).filter(lambda f: '.txt' in f).map(lambda f: './data/' + f)
for i, l in zip(images, labels): print(i, l)
```

```
./data/1.png ./data/1.txt
./data/2.png ./data/2.txt
./data/3.png ./data/3.txt
./data/4.png ./data/4.txt
./data/5.png ./data/5.txt
```

```Python
# Define functions for reading images and labels.
# Import statements must be inside a function to make multiprocessing work.
# This makes sure the name of the imported module is inside the local symbol table.
def read_image(path):
    '''
    Returns a numpy array.
    '''
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path))

def read_label(path):
    '''
    Returns an integer.
    '''
    with open(path, 'r') as f:
        return int(f.readline())
```

```Python
# The original data, which is a list of strings here, won't change after lazy_map() is called.
# If using map(), a list of strings will be transformed into a list of numpy arrays or integers.
# This is the key to solve the memory issue.
images = images.lazy_map(read_image)
labels = labels.lazy_map(read_label)
```

```Python
# Dataset.get() is a method to retrieve the stored data.
# We can see the mapping occurs when __getitem__() is invoked.
# But the stored data won't changed before and after invoking __getitem__().
for i in range(len(images)): print(images.get(i), type(images[i]))
for i in range(len(labels)): print(labels.get(i), type(labels[i]))
```

```
./data/1.png <class 'numpy.ndarray'>
./data/2.png <class 'numpy.ndarray'>
./data/3.png <class 'numpy.ndarray'>
./data/4.png <class 'numpy.ndarray'>
./data/5.png <class 'numpy.ndarray'>
./data/1.txt <class 'int'>
./data/2.txt <class 'int'>
./data/3.txt <class 'int'>
./data/4.txt <class 'int'>
./data/5.txt <class 'int'>
```

```Python
# Use two workers to read data.
# An error will occur if 'multiprocess' is not installed.
# Fix it by installing 'multiprocess' or not passing `parallel`.
images_loader, labels_loader = get_loader(images, labels, batch_size=2, parallel=2)
for X, y in zip(images_loader, labels_loader): print(len(X), len(y))
```

```
2 2
2 2
1 1
```
