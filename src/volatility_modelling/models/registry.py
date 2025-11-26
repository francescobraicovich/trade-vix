REGISTRY = {}
def register(name): 
    def deco(cls): REGISTRY[name] = cls; return cls
    return deco
def build(name, cfg): 
    return REGISTRY[name](cfg)
