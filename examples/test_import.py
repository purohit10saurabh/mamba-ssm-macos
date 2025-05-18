#!/usr/bin/env python

import sys

print("Testing mamba import options...")

# Try importing mamba directly
try:
    import mamba
    print(f"✅ Successfully imported 'mamba' directly")
    print(f"   Version: {mamba.__version__ if hasattr(mamba, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"❌ Failed to import 'mamba' directly: {e}")

# Try importing mamba_ssm but only access __version__
try:
    # Read the version without importing the whole package
    with open('mamba_ssm/__init__.py', 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                version = line.split('=')[1].strip(' "\'\n')
                break
    print(f"✅ Successfully found 'mamba_ssm' package")
    print(f"   Version: {version}")
except Exception as e:
    print(f"❌ Error checking 'mamba_ssm' package: {e}")

# Try importing core functional components by directly loading the file
try:
    # Add the current directory to sys.path to enable direct imports
    sys.path.insert(0, '.')
    
    # Import selective_scan_interface directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "selective_scan_interface", 
        "mamba_ssm/ops/selective_scan_interface.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check if selective_scan_fn exists
    if hasattr(module, 'selective_scan_fn'):
        print("✅ Successfully loaded 'selective_scan_fn' directly from file")
    else:
        print("❌ 'selective_scan_fn' not found in module")
except Exception as e:
    print(f"❌ Error loading selective_scan_interface: {e}")

print("\nImport test complete.") 