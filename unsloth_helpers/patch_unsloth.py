import sys
import os
import importlib.abc
import importlib.util

class LocalLlamaFinder(importlib.abc.MetaPathFinder):
    """
    A MetaPathFinder that intercepts any attempt to import 'unsloth.models.llama'
    and returns a spec pointing at our local 'unsloth_helpers/llama.py' which implements the summary embeddings.
    """
    def find_spec(self, fullname, path, target=None):
        # Only intercept exactly "unsloth.models.llama"
        if fullname == "unsloth.models.llama":
            # Compute the path to your local llama.py
            local_path = os.path.join(os.getcwd(), "unsloth_helpers", "llama.py")
            return importlib.util.spec_from_file_location(fullname, local_path)
        return None

# Insert our finder at the front of meta_path so it has top priority.
sys.meta_path.insert(0, LocalLlamaFinder())