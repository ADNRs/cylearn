import random
from copy import deepcopy
from .Dataset import Dataset


class _Cache:
    def __init__(self, size):
        if not isinstance(size, int):
            raise TypeError('`size` must be int')

        self.cache = [None for x in range(size)]
        self.count = 0

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, key):
        if not isinstance(key, (int, slice)):
            raise TypeError('`key` must be int or slice')
        return self.cache[key]

    def __setitem__(self, key, value):
        if not isinstance(key, (int, slice)):
            raise TypeError('`key` must be int or slice')

        if isinstance(key, slice):
            for data in self[key]:
                if data is None:
                    self.count += 1
        else:
            self.count += 1

        self.cache[key] = value

    def is_full(self):
        return self.count == len(self)


class Loader:
    '''
    Make a Dataset instance iterable on the batch level.

    `Loader` supports multiple features including prefetching, caching,
    and multiprocessing.

    Prefetching helps to accelerate the training process by enhancing
    latency and throughput. Imagining a model that runs on a GPU, using
    prefetching can achieve the parallelism between the model (on GPU)
    and data loading (on CPU).

    Caching means all data will be stored in the memory after the first
    read. If data is located in secondary storage such as HDD, caching
    could be used to speed up data loading when the capacity of
    available memory is enough.

    Multiprocessing imporves the performance on IO-bound tasks. If set
    properly, this will utilize all available CPU threads to load data.
    This feature requires 'multiprocess' which is a fork of Python
    'multiprocessing'.

    Attributes
    ----------
    dataset : Dataset
    batch_size : int
    shuffle : bool
    drop_last : bool
    prefetch : int
    parallel : int
    persistent : bool

    '''

    def __init__(self, dataset, batch_size, *,
                 shuffle=True, drop_last=False, prefetch=0,
                 parallel=1, persistent=True, enable_cache=False):
        '''
        Constructor.

        Parameters
        ----------
        dataset : Dataset
        batch_size : int
        shuffle : bool=True
            Keyword only.
        drop_last : bool, default=False
            Keyword only.
            If Ture, the last batch will be discarded if its length is
            less than `batch_size`.
        prefetch : int, default=0
            Keyword only.
            Indicating how many batches to be prefetched. `prefetch`
            should be greater than or equal to 0.
        parallel : int, default=1
            Keyword only.
            The number of workers. `parallel` should be greater than or
            equal to 1.
        persistent : bool, default=True
            Keyword only.
            If True, workers will not be closed after finishing their
            jobs.
        enable_cache : bool, default=False
            Keyword only.
            Use with caution, out of memory may happen.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            `dataset` is not an Dataset instance.
            `batch_size` is not an integer.
            `shuffle` is not a bool.
            `drop_last` is not a bool.
            `prefetch` is not an integer.
            `parallel` is not an integer.
            `persistent` is not a bool.
            `enable_cache` is not a bool.
        ValueError
            `parallel` is less than 1.
            `prefetch` is less than 0.
            `parallel` is greater than 1 but 'multiprocess' is not
            installed.

        '''
        if not isinstance(dataset, Dataset):
            raise TypeError('type of `dataset` must be Dataset')
        if not isinstance(batch_size, int):
            raise TypeError('type of `batch_size` must be int')
        if not isinstance(shuffle, bool):
            raise TypeError('type of `shuffle` must be bool')
        if not isinstance(drop_last, bool):
            raise TypeError('type of `drop_last` must be bool')
        if not isinstance(prefetch, int):
            raise TypeError('type of `prefetch` must be int')
        if not isinstance(parallel, int):
            raise TypeError('type of `parallel` must be int')
        if not isinstance(persistent, bool):
            raise TypeError('type of `persistent` must be bool')
        if not isinstance(enable_cache, bool):
            raise TypeError('type of `enable_cache` must be bool')
        if parallel < 1:
            raise ValueError('`parallel` must be greater than or equal to 1')
        if prefetch < 0:
            raise ValueError('`prefetch` must be greater than or equal to 0')
        if parallel > 1:
            try:
                import multiprocess
            except ModuleNotFoundError as e:
                raise ValueError(
                    '`parallel` can only be 1 if \'multiprocess\' is not '
                    'installed. Type `pip install multiprocess` in the '
                    'command prompt to fix this issue.'
                ) from e
            else:
                # expose multiprocess to the global symbol table
                globals()['multiprocess'] = locals()['multiprocess']

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch = prefetch
        self.persistent = persistent

        # no need to create workers when all data is in memory
        self.parallel = parallel if self.dataset._mapper is not None else 1

        # no need to use cache when all data is in memory
        self._cache = _Cache(len(dataset)) \
            if enable_cache and self.dataset._mapper is not None else None

    def __len__(self):
        num_batch = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            num_batch += 1
        return num_batch

    def __iter__(self):
        if self.persistent and self.parallel > 1:
            self._pool = multiprocess.Pool(self.parallel)
        else:
            self._pool = None

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            for i in range(len(self.dataset) - 1, 0, -1):
                j = random.randint(0, i)
                indices[i], indices[j] = indices[j], indices[i]

        batch_indices_iter = iter(self._batch_indices_helper(indices))
        if self.prefetch:
            while True:
                prefetch_buffer = list()
                try:
                    for i in range(self.prefetch):
                        prefetch_buffer.append(
                            self._get_batch(next(batch_indices_iter))
                        )
                except StopIteration:
                    break
                finally:
                    for batch in prefetch_buffer:
                        yield batch
        else:
            while True:
                try:
                    yield self._get_batch(next(batch_indices_iter))
                except StopIteration:
                    break

        if (self._pool is not None
            and self._cache is not None
                and self._cache.is_full()):
            self._pool.close()

    def _batch_indices_helper(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices

    def _get_batch(self, batch_indices):
        if self._cache is None:
            return self._get_batch_without_cache(batch_indices)
        else:
            return self._get_batch_with_cache(batch_indices)

    def _get_batch_without_cache(self, batch_indices):
        if self._pool is None:
            return [self.dataset[i] for i in batch_indices]
        else:
            data = [self.dataset.get(i) for i in batch_indices]
            if self.dataset._mapper is None:
                return data
            else:
                if self.persistent:
                    return self._pool.map(self.dataset._mapper, data)
                else:
                    self._pool = multiprocess.Pool(parallel)
                    data = self._pool.map(self.dataset._mapper, data)
                    self._pool.close()
                    return data

    def _get_batch_with_cache(self, batch_indices):
        not_in_cache_indices = \
            [i for i in batch_indices if self._cache[i] is None]

        if len(not_in_cache_indices):
            data = self._get_batch_without_cache(not_in_cache_indices)
            for i, data in zip(not_in_cache_indices, data):
                self._cache[i] = deepcopy(data)

        return [self._cache[i] for i in batch_indices]


def get_loader(dataset1, dataset2=None, batch_size=None, **kwargs):
    '''
    Help consturcting loader(s) in a more convenient way.

    Depending on different requirements, this function can be called in
    these two different arugemnt sets:
        1. get_loader(dataset1, 128, **kwargs) or
        2. get_loader(dataset1, dataset2, 128, **kwargs).

    When `dataset2` is given, make sure it has the same length of
    `dataset1`.

    If `dataset1` or `dataset2` is a list-like object, it will be used
    to construct a Dataset instance that contains itself before passing
    to the constructor of `Loader`.

    Parameters
    ----------
    dataset1 : Dataset or list-like
    dataset2 : Dataset, list-like or int
        If `dataset2` is an integer, `batch_size` should not be passed.
    batch_size: int, optional
    **kwargs : dict, optional
        Check the docstring of `Loader.__init__()`.

    Returns
    -------
    loader1 : Loader
    loader2 : Loader, conditional
        `loader2` is returned only if `dataset2` is a Dataset instance.

    Raises
    ------
    TypeError
        `dataset1` is not a Dataset instance.
        `dataset2` is not a Dataset instance or an integer.
        `batch_size` is not an integer.
    AssertionError
        If `dataset2` is given and the length of `dataset2` is
        inconsitent with `dataset1`.

    Notes
    -----
    Since Python does not support function overloading, the flexibility
    of this function is accomplished by lots of checks. This may cause
    the error message out of keeping with the caller's intention.

    '''
    if not isinstance(dataset1, Dataset):
        try:
            dataset1 = Dataset(dataset1)
        except TypeError:
            raise TypeError('type of `dataset1` must be Dataset or list-like')
    if not isinstance(dataset2, (Dataset, int, type(None))):
        try:
            dataset2 = Dataset(dataset2)
        except TypeError:
            raise TypeError(
                'type of `dataset2` must be Dataset, list-like or int'
            )
    if not isinstance(batch_size, (int, type(None))):
        if not isinstance(dataset2, (Dataset, type(None))):
            raise TypeError(
                'type of `dataset2` must be Dataset or list-like'
            )
        else:
            raise TypeError('type of `batch_size` must be int')

    # handle get_loader(dataset1)
    if (dataset2 is None and batch_size is None):
        raise TypeError('`batch_size` is not provided')

    # handle get_loader(dataset1, dataset2)
    if (isinstance(dataset1, Dataset)
        and isinstance(dataset2, Dataset)
            and batch_size is None):
        raise TypeError('`batch_size` is not provided')

    # handle get_loader(dataset1, _, int), _ is not a Dataset instance
    if (batch_size is not None
            and not isinstance(dataset2, Dataset)
            and dataset2 is not None):
        raise TypeError('type of `dataset2` must be Dataset or list-like')

    if isinstance(dataset2, int) and batch_size is None:
        dataset2, batch_size = batch_size, dataset2

    if dataset2 is None:
        return Loader(dataset1, batch_size, **kwargs)
    else:
        assert len(dataset1) == len(dataset2), \
            'inconsistent length between `dataset1` and `dataset2`'
        return Loader(dataset1, batch_size, **kwargs), \
            Loader(dataset2, batch_size, **kwargs)
