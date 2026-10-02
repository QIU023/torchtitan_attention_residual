import sys
extra = sys.argv[1]
sys.path.append(extra)
import numpy, torch  # global versions first
import pytest
print("numpy", numpy.__version__, numpy.__file__[:60])
sys.exit(pytest.main(sys.argv[2:]))
