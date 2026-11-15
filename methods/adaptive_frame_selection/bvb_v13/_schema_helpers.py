import math

def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>1e6:raise ValueError('Expected bounded finite number')
    return float(value)

def vector(value,positive=False):
    if not isinstance(value,list) or len(value)!=3:raise ValueError('Expected xyz vector')
    result=[number(x) for x in value]
    if positive and any(x<=1e-6 for x in result):raise ValueError('Dimensions must be positive')
    return result

def rotation(value):return [0.,0.,number(value)] if type(value) in (int,float) else vector(value)

def integer(value,lo,hi):
    if type(value) is not int or not lo<=value<=hi:raise ValueError('Index outside supplied data')
    return value
