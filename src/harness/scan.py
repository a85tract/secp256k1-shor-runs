"""Static source check of a candidate: it may reach the harness only through `h` and the objects `h` returns.
Flags imports outside an allowlist, introspection builtins and any attribute access to a name starting with
'_' (harness internals such as point labels live there). A flagged candidate's results are untrusted."""
import ast

ALLOWED_IMPORTS = {'math', 'cmath', 'fractions', 'decimal', 'itertools', 'functools', 'collections', 'heapq',
                   'bisect', 'operator', 'random', 'statistics', 'numpy', 'typing', 'dataclasses'}
FORBIDDEN_NAMES = {'eval', 'exec', 'compile', 'open', '__import__', 'getattr', 'setattr', 'delattr', 'globals',
                   'locals', 'vars', 'breakpoint', 'input', 'type', '__builtins__', 'object'}


def scan(path):
    src = open(path).read()
    tree = ast.parse(src, path)
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split('.')[0] not in ALLOWED_IMPORTS:
                    issues.append((node.lineno, 'import ' + a.name))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] not in ALLOWED_IMPORTS or node.level:
                issues.append((node.lineno, 'from %s import' % node.module))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            issues.append((node.lineno, 'name ' + node.id))
        elif isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            issues.append((node.lineno, 'attribute .' + node.attr))
    return issues


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        print(p, scan(p) or 'clean')
