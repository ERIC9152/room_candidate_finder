import torchvision.models as _m, functools as _f
def _wrap(fn):
    @_f.wraps(fn)
    def g(*a, pretrained=False, **k):
        k.pop('weights', None); return fn(*a, weights=None, **k)
    return g
for _n in ('resnet18','resnet34','resnet50','resnet101','resnet152','densenet121','densenet169'):
    setattr(_m, _n, _wrap(getattr(_m, _n)))
