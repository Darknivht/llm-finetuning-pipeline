#!/usr/bin/env python3
"""
Test runner script for the LLM Fine-Tuning Pipeline.
Provides convenient commands to run different types of tests.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(command, description=""):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def install_playwright():
    """Install Playwright browsers if needed."""
    print("\n🎭 Installing Playwright browsers...")
    commands = [
        "playwright install chromium",
        "playwright install-deps chromium"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Installing Playwright dependencies"):
            return False
    
    return True


def run_unit_tests():
    """Run unit tests."""
    return run_command(
        "pytest tests/unit/ -v",
        "Running unit tests"
    )


def run_integration_tests():
    """Run integration tests."""
    return run_command(
        "pytest tests/integration/ -v -m integration",
        "Running integration tests"
    )


def run_e2e_tests():
    """Run E2E tests."""
    return run_command(
        "pytest tests/e2e/ -v --headed",
        "Running E2E tests"
    )


def run_all_tests():
    """Run all tests."""
    return run_command(
        "pytest tests/ -v",
        "Running all tests"
    )


def run_tests_with_coverage():
    """Run tests with coverage report."""
    commands = [
        "pip install coverage pytest-cov",
        "pytest tests/ --cov=backend --cov-report=html --cov-report=term"
    ]
    
    for cmd in commands:
        if not run_command(cmd, "Running tests with coverage"):
            return False
    
    print("\n📊 Coverage report generated in htmlcov/index.html")
    return True


def run_quick_tests():
    """Run quick tests (excluding slow ones)."""
    return run_command(
        'pytest tests/ -v -m "not slow"',
        "Running quick tests"
    )


def run_smoke_tests():
    """Run smoke tests to verify basic functionality."""
    return run_command(
        "pytest tests/unit/test_models.py::TestModelLoader::test_load_tokenizer_success -v",
        "Running smoke tests"
    )


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Test runner for LLM Fine-Tuning Pipeline")
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=[
            "unit", "integration", "e2e", "all", "coverage", 
            "quick", "smoke", "install-playwright"
        ],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--install-playwright",
        action="store_true",
        help="Install Playwright browsers before running E2E tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🧪 LLM Fine-Tuning Pipeline - Test Runner")
    print("=" * 50)
    
    # Install Playwright if requested
    if args.install_playwright or args.test_type == "install-playwright":
        if not install_playwright():
            print("❌ Failed to install Playwright")
            return 1
        
        if args.test_type == "install-playwright":
            return 0
    
    # Run tests based on type
    success = True
    
    if args.test_type == "unit":
        success = run_unit_tests()
    
    elif args.test_type == "integration":
        success = run_integration_tests()
    
    elif args.test_type == "e2e":
        # Install Playwright if not already done
        if not args.install_playwright:
            print("📋 E2E tests require Playwright browsers")
            response = input("Install Playwright browsers now? (y/n): ")
            if response.lower() in ['y', 'yes']:
                if not install_playwright():
                    return 1
            else:
                print("⚠️ Skipping E2E tests - Playwright not installed")
                return 0
        
        success = run_e2e_tests()
    
    elif args.test_type == "all":
        print("🔄 Running all test suites...")
        
        success &= run_unit_tests()
        success &= run_integration_tests()
        
        # Ask about E2E tests
        response = input("\n🎭 Run E2E tests? (requires Playwright, may take longer) (y/n): ")
        if response.lower() in ['y', 'yes']:
            if not install_playwright():
                print("⚠️ Skipping E2E tests - Playwright installation failed")
            else:
                success &= run_e2e_tests()
    
    elif args.test_type == "coverage":
        success = run_tests_with_coverage()
    
    elif args.test_type == "quick":
        success = run_quick_tests()
    
    elif args.test_type == "smoke":
        success = run_smoke_tests()
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())