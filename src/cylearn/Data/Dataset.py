import math
import random
from copy import deepcopy


class Dataset:
    '''
    Provide some utilities for list-like data.

    A Dataset instance should be regarded as an immutable object. Any
    operation to a Dataset instance does not affect the instance itself,
    it instead returns a new copy with the designated operation applied.

    Although it is not recommended, however, if user wants to modify the
    original data, this can be done by manipulating `data` directly.
    But note that `data` does not keep up to date if `lazy_map()` was
    performed only.

    Methods
    -------
    get(key)
    copy()
    map(func)
    lazy_map(func)
    reduce(func)
    filter(func)

    See Also
    --------
    lazy_map(func)

    '''

    def __init__(self, data):
        '''
        Constructor.

        Parameters
        ----------
        data : list-like

        Returns
        -------
        None

        Raises
        ------
        TypeError
            `data` is not a list-like object.
        '''
        try:
            iter(data)
            len(data)
        except TypeError:
            raise TypeError('`data` must be a list-like object')

        self._data = deepcopy(list(data))
        self._mapper = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if not isinstance(key, (int, slice)):
            raise TypeError('`key` must be int or slice')

        if self._mapper is None:
            return self.get(key)

        if isinstance(key, slice):
            return [self._mapper(val) for val in self.get(key)]
        else:
            return self._mapper(self.get(key))

    def get(self, key):
        '''
        Get an item directly from `data`.

        Parameters
        ----------
        i : int or slice

        Returns
        -------
        object
            An object stored in `data`.

        Raises
        ------
        TypeError
            `i` is not an integer or a slice object.

        '''
        if not isinstance(key, (int, slice)):
            raise TypeError('`key` must be int or slice')

        return deepcopy(self._data[key])

    def copy(self):
        return deepcopy(self)

    def map(self, func):
        '''
        Map all elements by the given function.

        Parameters
        ----------
        func : callable
            `func` should take an argument and return a value.

        Returns
        -------
        dataset : Dataset

        Raises
        ------
        TypeError
            `func` is not a callable.

        See Also
        --------
        lazy_map(func)

        Notes
        -----
        Any call to `lazy_map()` would be eagerly evaluated once `map()`
        is called.

        '''
        if not callable(func):
            raise TypeError('`func` must be a callable')

        dataset = self.copy()
        dataset._update_mapper(func)
        dataset._data = dataset[:]
        dataset._mapper = None
        return dataset

    def lazy_map(self, func):
        '''
        Add a function for mapping all elements.

        `func` will be merged into `mapper` after calling `lazy_map`.
        `mapper` is a composition of functions and will only be applied
        to an element when it is being accessed via __getitem__().
        `data` will stay intact on both original or returned Dataset
        instance.

        `lazy_map()` uses lazy evaluation to delay calculations until
        needed. If `func` may cause mapped results to occupy a large
        amount of memory space, `lazy_map()` would be a good choice to
        trade the memory space with computation time.

        Since lazy evaluation is used, it is recommended that use
        __getitem__() to check that if there is any problem when
        mapping with `mapper`.

        Parameters
        ----------
        func : callable
            `func` should take an argument and return a value.

        Returns
        -------
        dataset : Dataset

        Raises
        ------
        TypeError
            `func` is not a callable.

        See Also
        --------
        __getitem__(i)
        map(func)

        '''
        if not callable(func):
            raise TypeError('`func` must be a callable')

        dataset = self.copy()
        dataset._update_mapper(func)
        return dataset

    def reduce(self, func):
        '''
        Reduce all elements into one by the given function.

        Parameters
        ----------
        func : callable
            `func` should take two arguments and return a value.

        Returns
        -------
        val : object
            Type of this object depends on the first element of `data`
            and `func`.

        Raises
        ------
        TypeError
            `func` is not a callable.

        Notes
        -----
        When `data` has only one element, `reduce()` will just return
        its copy even `func` does not fit the requirement.

        '''
        if not callable(func):
            raise TypeError('`func` must be a callable')

        val = deepcopy(self[0])
        for i in range(1, len(self)):
            val = func(val, self[i])
        return val

    def filter(self, func):
        '''
        Filter the data by the given function.

        Parameters
        ----------
        func : callable
            `func` should take two arguments and return a value which
            can be interpreted as a truth value.

        Returns
        -------
        Dataset

        Raises
        ------
        TypeError
            `func` is not a callable.

        Notes
        -----
        The evaluation strategy of `filter()` could only be eager. Use
        this with caution if user has called `lazy_map()` for some
        concerns before.

        '''
        if not callable(func):
            raise TypeError('`func` must be a callable')

        data = list()
        for i in range(len(self)):
            val = self[i]
            if func(val):
                data.append(deepcopy(val))
        return type(self)(data)

    def _update_mapper(self, func):
        if self._mapper is None:
            self._mapper = func
        else:
            def wrapper(x, mapper=self._mapper): return func(mapper(x))
            self._mapper = wrapper


def shuffle(dataset1, dataset2=None, seed=None):
    '''
    Perform Fisher–Yates shuffle.

    Parameters
    ----------
    dataset1 : Dataset
    dataset2 : Dataset, optional
        `dataset2` will be shuffled in unison with `dataset1`.
    seed : int, optional
        A number for setting the Python built-in random number
        generator.

    Returns
    -------
    dataset1 : Dataset
    dataset2 : Dataset, conditional
        `dataset2` is returned only if `dataset2` is passed in.

    Raises
    ------
    TypeError
        `dataset1` is not a Dataset instance.
        `dataset2` is not a Dataset instance.
        `seed` is not an integer.
    AssertionError
        If the length of `dataset2` is inconsistent with the length of
        `dataset1`.

    '''
    if not isinstance(dataset1, Dataset):
        raise TypeError('type of `dataset1` must be Dataset')
    if not isinstance(dataset2, (Dataset, type(None))):
        raise TypeError('type of `dataset2` must be Dataset')
    if not isinstance(seed, (int, type(None))):
        raise TypeError('type of `seed` must be int')
    else:
        random.seed(seed)

    dataset1 = dataset1.copy()
    if isinstance(dataset2, Dataset):
        dataset2 = dataset2.copy()

    if dataset2 is not None:
        assert len(dataset1) == len(dataset2), \
            'inconsistent length between `dataset1` and `dataset2`'

        for i in range(len(dataset1) - 1, 0, -1):
            j = random.randint(0, i)
            dataset1._data[i], dataset1._data[j] = \
                dataset1._data[j], dataset1._data[i]
            dataset2._data[i], dataset2._data[j] = \
                dataset2._data[j], dataset2._data[i]
    else:
        for i in range(len(dataset1) - 1, 0, -1):
            j = random.randint(0, i)
            dataset1._data[i], dataset1._data[j] = \
                dataset1._data[j], dataset1._data[i]

    if dataset2 is None:
        return dataset1
    else:
        return dataset1, dataset2


def split(dataset, ratio):
    '''
    Split a dataset into two according to a given ratio.

    Parameters
    ----------
    dataset : Dataset
    ratio : float
        `ratio` should lies in (0, 1).

    Returns
    -------
    dataset1 : Dataset
        `dataset1` contains the first `ratio` percent data of `dataset`.
    dataset2 : Dataset
        `dataset2` contains the remainder.

    Raises
    ------
    TypeError
        `dataset` is not a Dataset instance.
        `ratio` is not a float.
    ValueError
        `ratio` is greater than 1 or less than 0.

    '''
    if not isinstance(dataset, Dataset):
        raise TypeError('type of `dataset` must be Dataset')
    if not isinstance(ratio, float):
        raise TypeError('type of `ratio` must be float')
    if 0 >= ratio or ratio >= 1:
        raise ValueError('`ratio` should lies in (0, 1)')

    mid = math.ceil(ratio * len(dataset))
    dataset1 = type(dataset)(dataset._data[:mid])
    dataset2 = type(dataset)(dataset._data[mid:])
    dataset1._mapper = dataset._mapper
    dataset2._mapper = dataset._mapper
    return dataset1, dataset2
