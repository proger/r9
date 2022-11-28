#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

from torch import is_tensor

import sys

def exception_hook(exc_type, exc_value, tb):

    tb = tb.tb_next

    while tb:
        #x=tb.tb_frame.f_code
        # for field in dir(x):
            # print(f'@@@ {field} {getattr(x, field)}')

        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        line_no = tb.tb_lineno
        print(f'  File "{filename}", line {line_no}, in {name}')
        print(open(filename, 'r').readlines()[line_no-1], end='')

        local_vars = tb.tb_frame.f_locals

        for n,v in local_vars.items():
            if is_tensor(v):
                print(f'  {n} -> {v.size()}:{v.dtype}:{v.device}')
            else:
                print(f'  {n} -> {v}')

        tb = tb.tb_next

    print(f'{exc_type.__name__}: {exc_value}')

sys.excepthook = exception_hook

######################################################################

if __name__ == '__main__':

    import torch

    def dummy(a,b):
        print(a@b)

    def blah(a,b):
        c=b+b
        dummy(a,c)

    m=torch.randn(2,3)
    x=torch.randn(3)
    blah(m,x)
    blah(x,m)
