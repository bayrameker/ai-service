#!/usr/bin/env python3
"""
Test runner script for the AI service
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for AI service")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    
    args = parser.parse_args()
    
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent
    os.chdir(backend_dir)
    
    # Base pytest command
    pytest_cmd = "python -m pytest"
    
    # Add options
    if args.verbose:
        pytest_cmd += " -v"
    
    if args.parallel:
        pytest_cmd += " -n auto"
    
    if args.fast:
        pytest_cmd += " -m 'not slow'"
    
    if args.coverage:
        pytest_cmd += " --cov=app --cov-report=html --cov-report=term-missing"
    
    # Determine which tests to run
    test_commands = []
    
    if args.unit:
        cmd = f"{pytest_cmd} tests/test_*.py -m 'not integration and not performance'"
        test_commands.append((cmd, "Unit Tests"))
    
    elif args.integration:
        cmd = f"{pytest_cmd} tests/test_integration.py"
        test_commands.append((cmd, "Integration Tests"))
    
    elif args.performance:
        cmd = f"{pytest_cmd} tests/test_performance.py"
        test_commands.append((cmd, "Performance Tests"))
    
    else:
        # Run all tests
        if not args.fast:
            test_commands.extend([
                (f"{pytest_cmd} tests/test_llm_providers.py", "LLM Provider Tests"),
                (f"{pytest_cmd} tests/test_agents.py", "Agent Tests"),
                (f"{pytest_cmd} tests/test_memory.py", "Memory Tests"),
                (f"{pytest_cmd} tests/test_integration.py", "Integration Tests"),
                (f"{pytest_cmd} tests/test_performance.py", "Performance Tests")
            ])
        else:
            test_commands.extend([
                (f"{pytest_cmd} tests/test_llm_providers.py -m 'not slow'", "LLM Provider Tests (Fast)"),
                (f"{pytest_cmd} tests/test_agents.py -m 'not slow'", "Agent Tests (Fast)"),
                (f"{pytest_cmd} tests/test_memory.py -m 'not slow'", "Memory Tests (Fast)"),
                (f"{pytest_cmd} tests/test_integration.py -m 'not slow'", "Integration Tests (Fast)")
            ])
    
    # Run tests
    all_passed = True
    results = []
    
    for command, description in test_commands:
        success = run_command(command, description)
        results.append((description, success))
        if not success:
            all_passed = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{description}: {status}")
    
    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    
    # Generate coverage report if requested
    if args.coverage and all_passed:
        print(f"\n{'='*60}")
        print("COVERAGE REPORT")
        print(f"{'='*60}")
        
        # Open coverage report
        coverage_file = backend_dir / "htmlcov" / "index.html"
        if coverage_file.exists():
            print(f"Coverage report generated: {coverage_file}")
            print("Open in browser to view detailed coverage report")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
