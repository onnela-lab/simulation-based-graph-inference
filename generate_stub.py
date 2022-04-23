import argparse
import importlib
import inspect
import re
import textwrap
import typing

INDENT = "    "
TQUOTE = '"""'
NEWLINE = "\n"


class Literal:
    """
    Object whose `repr` returns its name or maps child types to names.
    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        if isinstance(self.name, dict):
            raise ValueError
        return self.name

    def __getattr__(self, name: str) -> 'Literal':
        if isinstance(self.name, dict):
            return Literal(self.name[name])
        return Literal(f"{self.name}.{name}")


def redent(text, indent):
    return textwrap.indent(textwrap.dedent(text), indent)


def get_stub(func: typing.Callable, typedefs: typing.Mapping[str, typing.Any] = None) \
        -> typing.Callable:
    """
    Generate a stub by executing the cython function signature.

    Args:
        func: Function to generate a stub for.
        typedefs: Mapping from names to types to account for `ctypedef`s.

    Returns:
        stub: Function stub.
    """
    signature, *lines = func.__doc__.split("\n", 1)
    # Substitute duplicate type hints.
    signature = re.sub(r"[\w:]+ (?P<name>\w+): (?P<type>[\w.]+)", r"\g<name>: \g<type>", signature)
    # Substitute C-style type hints.
    signature = re.sub(r"(?P<type>[\w+:]\w+)\s+(?P<name>\w+)(?P<sep>[,)])",
                       r"\g<name>: \g<type>\g<sep>", signature)
    # Replace leading parent type specifiers for methods.
    signature = re.sub(r"^(?:\w+\.)+", "", signature)
    # Execute and retrieve the stub.
    locals().update(typedefs or {})
    exec(f'def {signature}: ...')
    stub = locals()[func.__name__]
    stub.__doc__ = textwrap.dedent("\n".join(lines)).strip()
    return stub


def format_stub(stub, include_doc: bool = True) -> str:
    """
    Format a function stub.

    Args:
        stub: Stub to format.
        include_doc: Whether to include the docstring.

    Returns:
        formatted: Formatted function stub.
    """
    lines = [
        f"def {stub.__name__}{inspect.signature(stub)}:",
    ]
    if include_doc:
        lines.extend(get_docstring_lines(stub))
    lines.append(INDENT + "...")
    return NEWLINE.join(lines)


def get_docstring_lines(obj) -> list[str]:
    if doc := textwrap.dedent(obj.__doc__).strip():
        return [
            f"{INDENT}r{TQUOTE}",
            textwrap.indent(doc, INDENT),
            INDENT + TQUOTE,
        ]
    return []


def generate_stubs(obj, typedefs, sep="\n") -> str:
    lines = []

    if inspect.isclass(obj):
        parts = []
        parts.append(f"class {obj.__name__}:")
        init = get_stub(obj, typedefs)
        parts.extend(get_docstring_lines(init))
        lines.append(NEWLINE.join(parts))
        # Add __init__ to the members. We fake the name to ensure it's included in the stub.
        init.__name__ = '__init__'
        init.__doc__ = ''
        members = [('..init..', init)]
        # Get members to process.
        members.extend(inspect.getmembers(obj, inspect.isroutine))
        sep = "\n\n"
    elif inspect.isroutine(obj):
        if inspect._signature_is_builtin(obj):
            obj = get_stub(obj, typedefs)
        lines.append(format_stub(obj))
        members = []
    else:
        raise TypeError(obj)

    for name, member in members:
        if name.startswith('__'):
            continue
        lines.append(textwrap.indent(generate_stubs(member, typedefs, sep), INDENT))

    return sep.join(lines)


def __main__(args: list[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('module', help='module to generate stubs for')
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    members = inspect.getmembers(module, lambda x: (inspect.isclass(x) or inspect.isroutine(x)) and
                                 not getattr(x, '__name__', '__').startswith('__'))
    # Add a header if available.
    lines = []
    try:
        lines.append(textwrap.dedent(module.__PYI_HEADER).strip())
    except AttributeError:
        pass

    # Generate the stub.
    try:
        typedefs = module.__PYI_TYPEDEFS
        typedefs = {key: Literal(value) for key, value in typedefs.items()}
    except AttributeError:
        typedefs = {}
    lines.extend(generate_stubs(member, typedefs) for _, member in members)
    print("\n\n\n".join(lines))


if __name__ == "__main__":
    __main__()
