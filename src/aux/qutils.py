def attrsetter(*items):
    def resolve_attr(obj, attr):
        attrs = attr.split(".")
        head = attrs[:-1]
        tail = attrs[-1]

        for name in head:
            obj = getattr(obj, name)
        return obj, tail

    def g(obj, val):
        for attr in items:
            resolved_obj, resolved_attr = resolve_attr(obj, attr)
            if resolved_attr == "bias":
                setattr(resolved_obj, resolved_attr, val.bias)
            else:
                setattr(resolved_obj, resolved_attr, val)

    return g

def is_biased(module) -> bool:
    return getattr(module, "bias", None) is not None