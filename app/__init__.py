import os
import time

# Ops missing on Apple's MPS backend run on the CPU instead of failing.
# Read when torch loads, so it is set here first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# When the server process began, before torch and the rest load: the startup cache scan's
# MATRICXON_START_TIMEOUT counts from here (app.dependencies).
STARTED_AT = time.monotonic()
