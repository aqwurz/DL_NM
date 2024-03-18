#!/usr/bin/env python3

import functools
import inspect


# the following code is from
# https://shwina.github.io/cython-testing/
def cytest(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        return func(*bound.args, **bound.kwargs)
    return wrapped
