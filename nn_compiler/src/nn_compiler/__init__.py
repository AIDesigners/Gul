
import imp

modules = set(["numpy", "data_structurews"])

for m in modules :
    try    : imp.find_module(m)
    except : print("Missing dependency: {:s}".format(str(m)))
    