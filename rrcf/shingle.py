from collections import deque

def shingle(sequence, size):
    """
    Generator that yields shingles (a rolling window) of a given size.

    Parameters
    ----------
    sequence : iterable
               Sequence to be shingled
    size : int
           size of shingle (window)
    """
    iterator = iter(sequence)
    init = (next(iterator) for _ in range(size))
    window = deque(init, maxlen=size)
    if len(window) < size:
        raise IndexError('Sequence smaller than window size')
    yield window
    for elem in iterator:
        window.append(elem)
        yield window
