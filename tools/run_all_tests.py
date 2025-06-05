#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from tests.unit.test_generation_macos import run_generation_tests
from tests.unit.test_mamba2_macos import run_mamba2_tests
from tests.unit.test_mamba_macos import run_mamba_tests


def run_all_tests():
    print("🚀 Starting comprehensive test suite...")
    
    test_modules = [
        ("Mamba macOS Tests", run_mamba_tests),
        ("Mamba2 macOS Tests", run_mamba2_tests),
        ("Generation macOS Tests", run_generation_tests)
    ]
    
    results = {}
    
    for test_name, test_func in test_modules:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name}: {status}")
            
        except Exception as e:
            print(f"❌ {test_name} execution error: {e}")
            results[test_name] = False
    
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test modules passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed")
        return False

def main():
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 