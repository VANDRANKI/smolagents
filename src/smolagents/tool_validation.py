"""Static validation for :class:`~smolagents.tools.Tool` subclasses.

This module provides AST-based checks that run when a ``Tool`` subclass is
defined or loaded.  The checks catch common mistakes early — before any model
call is made — and surface actionable error messages.

Validation rules enforced by :func:`validate_tool_attributes`:

1. **Class attributes** must be simple literals (strings, numbers, dicts,
   lists, sets).  Complex attributes (e.g. object instances, function calls)
   must be created inside ``__init__`` instead.
2. **``__init__`` parameters** must all have default values so that the tool
   can be reconstructed from its class definition alone (parameters chosen at
   init time cannot be recovered when serialising/deserialising the tool).
3. **All method bodies** must be self-contained: every name they reference must
   be either a builtin, an import, a class attribute, a function argument, or
   a locally assigned variable.  Local imports (relative imports that reference
   files rather than installed packages) are also rejected.
"""

import ast
import builtins
from itertools import zip_longest

from .utils import BASE_BUILTIN_MODULES, get_source, is_valid_name


_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    """
    AST visitor that validates a single method body for two properties:

    1. **No undefined names** — every name used in the method must be resolvable
       as a builtin, a module-level import, a class attribute, a function
       argument, or a locally-assigned variable.
    2. **No local imports** — ``import`` statements inside the method may only
       reference installed packages (e.g. ``import numpy``), never relative
       paths or project-local modules.

    Errors are collected in :attr:`errors` rather than raised immediately so
    that all problems in a method are reported at once.

    Args:
        class_attributes: Set of attribute names defined at class level; these
            are always considered defined inside any method.
        check_imports: If ``True`` (default), flag imports from local modules.
            Set to ``False`` to skip the import-locality check.
    """

    def __init__(self, class_attributes: set[str], check_imports: bool = True) -> None:
        self.undefined_names: set[str] = set()
        self.imports: dict[str, str] = {}
        self.from_imports: dict[str, tuple[str, str]] = {}
        self.assigned_names: set[str] = set()
        self.arg_names: set[str] = set()
        self.class_attributes: set[str] = class_attributes
        self.errors: list[str] = []
        self.check_imports: bool = check_imports
        self.typing_names: set[str] = {"Any"}
        self.defined_classes: set[str] = set()

    def visit_arguments(self, node: ast.arguments) -> None:
        """Collect all formal parameter names from a function signature."""
        self.arg_names = {arg.arg for arg in node.args}
        if node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        if node.vararg:
            self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node: ast.Import) -> None:
        """Record ``import X`` / ``import X as Y`` statements."""
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record ``from X import Y`` / ``from X import Y as Z`` statements."""
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track plain assignment targets as locally-defined names."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.visit(node.value)

    def visit_With(self, node: ast.With) -> None:
        """Track context-manager aliases (the ``y`` in ``with X as y``)."""
        for item in node.items:
            if item.optional_vars:  # This is the 'y' in 'with X as y'
                if isinstance(item.optional_vars, ast.Name):
                    self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Track exception aliases (the ``e`` in ``except Exception as e``)."""
        if node.name:  # This is the 'e' in 'except Exception as e'
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments (e.g. ``x: int = 0``)."""
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        """Track the loop variable in ``for`` statements."""
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators: list[ast.comprehension]) -> None:
        """Register the iteration variable(s) for each generator in a comprehension."""
        for generator in generators:
            if isinstance(generator.target, ast.Name):
                self.assigned_names.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Track variables bound by list comprehension generators."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Track variables bound by dict comprehension generators."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Track variables bound by set comprehension generators."""
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Skip ``self.xxx`` attribute accesses — they are always valid."""
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track inline class definitions so their names resolve correctly."""
        self.defined_classes.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Flag any name load that cannot be resolved in the current scope.

        A name is considered defined if it appears in any of: Python builtins,
        :data:`BASE_BUILTIN_MODULES`, function arguments, ``self``, class
        attributes, imports recorded by :meth:`visit_Import` /
        :meth:`visit_ImportFrom`, locally-assigned names, known typing names,
        or inline class definitions.
        """
        if isinstance(node.ctx, ast.Load):
            if not (
                node.id in _BUILTIN_NAMES
                or node.id in BASE_BUILTIN_MODULES
                or node.id in self.arg_names
                or node.id == "self"
                or node.id in self.class_attributes
                or node.id in self.imports
                or node.id in self.from_imports
                or node.id in self.assigned_names
                or node.id in self.typing_names
                or node.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node: ast.Call) -> None:
        """Flag any direct function call whose name cannot be resolved.

        Only bare name calls (e.g. ``foo()``) are checked here; attribute
        calls (e.g. ``obj.foo()``) are handled by :meth:`visit_Attribute`.
        The resolution rules are identical to :meth:`visit_Name`.
        """
        if isinstance(node.func, ast.Name):
            if not (
                node.func.id in _BUILTIN_NAMES
                or node.func.id in BASE_BUILTIN_MODULES
                or node.func.id in self.arg_names
                or node.func.id == "self"
                or node.func.id in self.class_attributes
                or node.func.id in self.imports
                or node.func.id in self.from_imports
                or node.func.id in self.assigned_names
                or node.func.id in self.defined_classes
            ):
                self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


def validate_tool_attributes(cls: type, check_imports: bool = True) -> None:
    """Validate that a ``Tool`` subclass follows the required structural patterns.

    This function parses the class source with the ``ast`` module and applies
    the checks listed below.  All errors found are collected and raised
    together as a single :class:`ValueError` so that contributors can fix
    everything in one pass.

    Checks performed:

    0. Every parameter of ``__init__`` (other than ``self``) must have a
       default value.  Arguments chosen at construction time cannot be
       recovered when the tool is serialised, so all important configuration
       must be stored as class-level attributes instead.
    1. **Class attributes** must be simple literals (``str``, ``int``,
       ``float``, ``bool``, ``None``, ``dict``, ``list``, or ``set``
       literals).  Anything more complex must go inside ``__init__``.
    2. **``__init__`` default values** must themselves be literals (same set
       as above); non-literal defaults (e.g. a function call) are rejected.
    3. **All method bodies** must only reference names that are resolvable at
       definition time (builtins, imports, class attributes, arguments, or
       locally-assigned variables).
    4. **Imports inside methods** must reference installed packages, not local
       project files.

    Args:
        cls: The ``Tool`` subclass to validate.  Its source must be
            retrievable via :func:`~smolagents.utils.get_source`.
        check_imports: If ``True`` (default), imports inside methods are
            checked to ensure they reference packages rather than local files.

    Raises:
        ValueError: If the class source cannot be parsed, is not a class
            definition, or if any validation rule is violated.  The error
            message lists all violations.
    """

    class ClassLevelChecker(ast.NodeVisitor):
        """First-pass visitor that collects class-level metadata.

        Attributes:
            imported_names: Names introduced by module-level imports.
            complex_attributes: Class attributes whose values are not simple
                literals (these should be moved to ``__init__``).
            class_attributes: All names assigned at class level.
            non_defaults: ``__init__`` parameters that have no default value.
            non_literal_defaults: ``__init__`` parameters whose default is not
                a literal (e.g. a function call result).
            in_method: ``True`` while visiting inside a method body; used to
                distinguish method-local assignments from class-level ones.
            invalid_attributes: Human-readable error strings for individual
                attribute violations (e.g. ``name`` must be a string).
        """

        def __init__(self) -> None:
            self.imported_names: set[str] = set()
            self.complex_attributes: set[str] = set()
            self.class_attributes: set[str] = set()
            self.non_defaults: set[str] = set()
            self.non_literal_defaults: set[str] = set()
            self.in_method: bool = False
            self.invalid_attributes: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == "__init__":
                self._check_init_function_parameters(node)
            old_context = self.in_method
            self.in_method = True
            self.generic_visit(node)
            self.in_method = old_context

        def visit_Assign(self, node: ast.Assign) -> None:
            if self.in_method:
                return
            # Track class attributes
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes.add(target.id)

            # Check if the assignment is more complex than simple literals
            if not all(isinstance(val, (ast.Constant, ast.Dict, ast.List, ast.Set)) for val in ast.walk(node.value)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.complex_attributes.add(target.id)

            # Check specific class attributes
            if getattr(node.targets[0], "id", "") == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(f"Class attribute 'name' must be a constant, found '{node.value}'")
                elif not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{node.value.value}'"
                    )
                elif not is_valid_name(node.value.value):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found '{node.value.value}'"
                    )

        def _check_init_function_parameters(self, node: ast.FunctionDef) -> None:
            """Validate that all ``__init__`` parameters have literal defaults."""
            # Check defaults in parameters
            for arg, default in reversed(list(zip_longest(reversed(node.args.args), reversed(node.args.defaults)))):
                if default is None:
                    if arg.arg != "self":
                        self.non_defaults.add(arg.arg)
                elif not isinstance(default, (ast.Constant, ast.Dict, ast.List, ast.Set)):
                    self.non_literal_defaults.add(arg.arg)

    class_level_checker = ClassLevelChecker()
    source = get_source(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    if not isinstance(class_node, ast.ClassDef):
        raise ValueError("Source code must define a class")
    class_level_checker.visit(class_node)

    errors: list[str] = []
    # Check invalid class attributes
    if class_level_checker.invalid_attributes:
        errors += class_level_checker.invalid_attributes
    if class_level_checker.complex_attributes:
        errors.append(
            f"Complex attributes should be defined in __init__, not as class attributes: "
            f"{', '.join(class_level_checker.complex_attributes)}"
        )
    if class_level_checker.non_defaults:
        errors.append(
            f"Parameters in __init__ must have default values, found required parameters: "
            f"{', '.join(class_level_checker.non_defaults)}"
        )
    if class_level_checker.non_literal_defaults:
        errors.append(
            f"Parameters in __init__ must have literal default values, found non-literal defaults: "
            f"{', '.join(class_level_checker.non_literal_defaults)}"
        )

    # Run checks on all methods
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            method_checker = MethodChecker(class_level_checker.class_attributes, check_imports=check_imports)
            method_checker.visit(node)
            errors += [f"- {node.name}: {error}" for error in method_checker.errors]

    if errors:
        raise ValueError(f"Tool validation failed for {cls.__name__}:\n" + "\n".join(errors))
    return
