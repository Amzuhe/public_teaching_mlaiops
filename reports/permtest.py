"""Lab 5 Task 2: try one upload through the adapter and print OK or the error code."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402

p = pathlib.Path("reports/permtest.txt")
p.write_text("lab5 permission test\n")
try:
    print("UPLOAD OK ->", get_adapter(config.load(strict=False)).upload(str(p), "lab5/permtest.txt"))
except Exception as e:
    print("UPLOAD FAILED:", type(e).__name__, getattr(e, "error_code", ""), str(e).splitlines()[0][:150])
