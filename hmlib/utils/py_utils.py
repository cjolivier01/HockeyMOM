import importlib
import pkgutil

from hmlib.log import get_logger


logger = get_logger(__name__)


def find_item_in_module(module, item_name):
    """
    Searches for an item by name in the given module and its submodules recursively.

    :param module: The module object to search in.
    :param item_name: The name of the item to search for.
    :return: The module path as a string if the item is found, otherwise None.
    """
    # Check if the item exists in the current module

    if isinstance(module, str):
        module = importlib.import_module(module)

    if hasattr(module, item_name):
        return module.__name__

    # If the module has a __path__ attribute, it's a package that can contain submodules
    if hasattr(module, "__path__"):
        for _, submodule_name, is_pkg in pkgutil.walk_packages(module.__path__):
            full_submodule_name = f"{module.__name__}.{submodule_name}"
            try:
                # Dynamically import the submodule
                submodule = importlib.import_module(full_submodule_name)
                # Recursively search in the submodule
                module_path = find_item_in_module(submodule, item_name)
                if module_path is not None:
                    return module_path
            except Exception as e:
                logger.exception("Error importing %s: %s", full_submodule_name, e)
                continue

    # Item not found in this module or its submodules
    return None


# Example usage
if __name__ == "__main__":
    pass  # Replace 'some_module' with
