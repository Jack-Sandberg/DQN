# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:07:55 2020

@author: Jack
"""
#From https://gist.github.com/jmfrank63/5fb9909a8e06c91dead9265cab2f33de

from collections import deque

class AsyncDeque(deque):
    def __init__(self, elements, maxlen):
        super().__init__(elements, maxlen)
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self:
            raise StopAsyncIteration
        element = self.popleft()
        return element