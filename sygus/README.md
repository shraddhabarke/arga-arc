edit `.venv/lib/python3.10/site-packages/sexpdata.py` line 199-201 to be:

```
obj = parse(string, **kwds)
# assert len(obj) == 1  # FIXME: raise an appropriate error
return obj
```