#!/usr/bin/env python3
"""
Unified build script for risk forecasting Cython modules.
Handles compilation of combined tvp_evt_core.pyx and quantile_regression_core.pyx.

Usage:
    python build.py                    # Build all modules
    python build.py --clean            # Clean and rebuild
    python build.py --test             # Build and test
    python build.py --benchmark        # Build and benchmark
    python build.py --verbose          # Verbose output
    python build.py --parallel         # Parallel compilation
    python build.py --profile          # Enable profiling
"""

import os
import sys
import subprocess
import argparse
import platform
import shutil
import time
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE):
    """Print colored text to console."""
    print(f"{color}{text}{Colors.END}")

def print_header(title: str):
    """Print formatted header."""
    print_colored("=" * 70, Colors.CYAN)
    print_colored(f"  {title}", Colors.BOLD + Colors.WHITE)
    print_colored("=" * 70, Colors.CYAN)

def print_step(step_num: int, description: str):
    """Print formatted step."""
    print_colored(f"\n{step_num}. {description}", Colors.BLUE + Colors.BOLD)

def print_success(message: str):
    """Print success message."""
    print_colored(f"✓ {message}", Colors.GREEN)

def print_warning(message: str):
    """Print warning message."""
    print_colored(f"⚠ {message}", Colors.YELLOW)

def print_error(message: str):
    """Print error message."""
    print_colored(f"✗ {message}", Colors.RED)

class BuildConfiguration:
    """Build configuration manager."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.is_64bit = platform.machine().endswith('64')
        
        # Detect compiler
        self.compiler = self._detect_compiler()
        
        # Build settings
        self.optimization_level = "O3"
        self.enable_openmp = self._check_openmp_support()
        self.enable_fast_math = True
        self.enable_native_arch = True
        
    def _detect_compiler(self) -> str:
        """Detect available C compiler."""
        compilers = ['gcc', 'clang', 'cl'] if self.platform == 'windows' else ['gcc', 'clang']
        
        for compiler in compilers:
            if shutil.which(compiler):
                return compiler
        
        return 'unknown'
    
    def _check_openmp_support(self) -> bool:
        """Check if OpenMP is available."""
        if self.platform == 'darwin':  # macOS
            return False  # OpenMP often problematic on macOS
        
        try:
            # Try to compile a simple OpenMP program
            test_code = '''
            #include <omp.h>
            int main() { return omp_get_num_threads(); }
            '''
            
            with open('test_openmp.c', 'w') as f:
                f.write(test_code)
            
            result = subprocess.run([
                self.compiler, '-fopenmp', 'test_openmp.c', '-o', 'test_openmp'
            ], capture_output=True, text=True)
            
            os.remove('test_openmp.c')
            if os.path.exists('test_openmp'):
                os.remove('test_openmp')
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def get_compile_args(self) -> List[str]:
        """Get compiler arguments based on platform and configuration."""
        args = []
        
        if self.platform == 'windows':
            # MSVC flags
            args.extend(['/O2', '/fp:fast'])
            if self.enable_openmp:
                args.append('/openmp')
        else:
            # GCC/Clang flags
            args.extend([f'-{self.optimization_level}'])
            
            if self.enable_fast_math:
                args.append('-ffast-math')
            
            if self.enable_native_arch and self.compiler != 'clang':
                args.extend(['-march=native', '-mtune=native'])
            
            if self.enable_openmp:
                args.append('-fopenmp')
            
            # Additional optimizations
            args.extend([
                '-funroll-loops',
                '-fomit-frame-pointer',
                '-ftree-vectorize'
            ])
        
        return args
    
    def get_link_args(self) -> List[str]:
        """Get linker arguments."""
        if self.platform == 'windows':
            return []
        else:
            args = [f'-{self.optimization_level}']
            if self.enable_openmp:
                args.append('-fopenmp')
            return args

class DependencyChecker:
    """Check and install required dependencies."""
    
    REQUIRED_PACKAGES = {
        'numpy': '1.19.0',
        'scipy': '1.7.0',
        'cython': '0.29.0',
        'setuptools': '40.0.0',
        'wheel': '0.30.0'
    }
    
    OPTIONAL_PACKAGES = {
        'pandas': '1.3.0',
        'polars': '0.18.0',
        'matplotlib': '3.3.0'  # For visualization
    }
    
    def check_python_version(self) -> bool:
        """Check if Python version is supported."""
        version = sys.version_info
        if version.major != 3 or version.minor < 7:
            print_error(f"Python 3.7+ required, found {version.major}.{version.minor}")
            return False
        
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_package(self, package_name: str, min_version: str = None) -> bool:
        """Check if a package is installed with minimum version."""
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False
            
            if min_version:
                try:
                    module = importlib.import_module(package_name)
                    if hasattr(module, '__version__'):
                        from packaging import version
                        return version.parse(module.__version__) >= version.parse(min_version)
                except ImportError:
                    pass
            
            return True
            
        except ImportError:
            return False
    
    def install_package(self, package_name: str, min_version: str = None) -> bool:
        """Install a package using pip."""
        package_spec = f"{package_name}>={min_version}" if min_version else package_name
        
        try:
            print(f"Installing {package_spec}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package_spec
            ], capture_output=True, text=True, check=True)
            
            print_success(f"Installed {package_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {package_name}: {e.stderr}")
            return False
    
    def check_and_install_dependencies(self, install_missing: bool = True) -> bool:
        """Check and optionally install all dependencies."""
        print_step(1, "Checking dependencies")
        
        if not self.check_python_version():
            return False
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package, min_version in self.REQUIRED_PACKAGES.items():
            if self.check_package(package, min_version):
                print_success(f"{package} >= {min_version}")
            else:
                missing_required.append((package, min_version))
                print_warning(f"{package} >= {min_version} - MISSING")
        
        # Check optional packages
        for package, min_version in self.OPTIONAL_PACKAGES.items():
            if self.check_package(package, min_version):
                print_success(f"{package} >= {min_version} (optional)")
            else:
                missing_optional.append((package, min_version))
                print_warning(f"{package} >= {min_version} (optional) - MISSING")
        
        # Install missing packages if requested
        if missing_required and install_missing:
            print("\nInstalling missing required packages...")
            for package, min_version in missing_required:
                if not self.install_package(package, min_version):
                    return False
        elif missing_required:
            print_error("Required packages missing. Run with --install-deps to install.")
            return False
        
        if missing_optional and install_missing:
            print("\nInstalling missing optional packages...")
            for package, min_version in missing_optional:
                self.install_package(package, min_version)  # Don't fail on optional
        
        return True

class CythonBuilder:
    """Handles Cython module compilation."""
    
    def __init__(self, config: BuildConfiguration, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.build_dir = Path('build')
        self.temp_dir = Path('temp_build')
        
    def clean_build_artifacts(self):
        """Clean previous build artifacts."""
        print_step(2, "Cleaning build artifacts")
        
        # Directories to remove
        dirs_to_clean = [
            'build', '__pycache__', 'temp_build',
            '.pytest_cache', 'htmlcov'
        ]
        
        for dir_name in dirs_to_clean:
            dir_path = Path(dir_name)
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print_success(f"Removed {dir_name}/")
        
        # Files to remove
        patterns_to_clean = [
            '*.c', '*.so', '*.pyd', '*.dll', '*.html',
            '*.pyc', '*.pyo', '*.egg-info'
        ]
        
        for pattern in patterns_to_clean:
            for file_path in Path('.').glob(pattern):
                if file_path.name != 'setup.py':  # Preserve setup.py
                    file_path.unlink()
                    print_success(f"Removed {file_path}")
    
    def create_setup_py(self, enable_profiling: bool = False) -> str:
        """Create setup.py with optimized configuration."""
        
        compiler_directives = {
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'nonecheck': False,
            'initializedcheck': False,
            'overflowcheck': False,
            'embedsignature': True,
            'language_level': 3
        }
        
        if enable_profiling:
            compiler_directives['profile'] = True
            compiler_directives['linetrace'] = True
        
        setup_content = f'''
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import os

# Build configuration
extra_compile_args = {self.config.get_compile_args()}
extra_link_args = {self.config.get_link_args()}

# Add platform-specific includes
include_dirs = [numpy.get_include()]

# Define extensions for combined modules
extensions = [
    Extension(
        name="risk_forecasting_core",
        sources=["analyzers/risk_engine.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c"
    )
]

# Compiler directives
compiler_directives = {compiler_directives}

setup(
    name="risk_forecasting_cython",
    version="1.0.0",
    description="High-performance risk forecasting with Cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate={self.verbose}
    ),
    zip_safe=False,
    python_requires=">=3.7"
)
'''
        
        setup_path = Path('setup_generated.py')
        setup_path.write_text(setup_content)
        
        return str(setup_path)
    
    def check_cython_source(self) -> bool:
        """Check if Cython source file exists."""
        source_file = Path('risk_engine.pyx')
        
        if not source_file.exists():
            source_file = Path('analyzers/risk_engine.pyx')
            if not source_file.exists():
                print_error(f"Cython source file not found: {source_file}")
                print("Expected file: risk_engine.pyx")
                print("This should contain the combined TVP-EVT and Quantile Regression code.")
                return False
        
        # Basic syntax check
        content = source_file.read_text()
        if len(content) < 1000:  # Sanity check
            print_warning("Cython source file seems very small")
        
        print_success(f"Found Cython source: {source_file} ({len(content)} chars)")
        return True
    
    def compile_modules(self, parallel: bool = False, enable_profiling: bool = False) -> bool:
        """Compile Cython modules."""
        print_step(3, "Compiling Cython modules")
        
        if not self.check_cython_source():
            return False
        
        # Create optimized setup.py
        setup_file = self.create_setup_py(enable_profiling)
        
        # Build command
        build_cmd = [
            sys.executable, setup_file,
            'build_ext', '--inplace'
        ]
        
        if parallel and self.config.platform != 'windows':
            # Add parallel compilation for Unix systems
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
            build_cmd.extend(['-j', str(n_jobs)])
        
        if self.verbose:
            build_cmd.append('--verbose')
        
        print(f"Build command: {' '.join(build_cmd)}")
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                build_cmd,
                capture_output=not self.verbose,
                text=True,
                check=True
            )
            
            compile_time = time.time() - start_time
            print_success(f"Compilation completed in {compile_time:.2f}s")
            
            # Check for generated files
            extensions = ['.so', '.pyd', '.dll']
            generated_files = []
            
            for ext in extensions:
                for file_path in Path('.').glob(f'*{ext}'):
                    generated_files.append(file_path)
            
            if generated_files:
                print_success("Generated files:")
                for file_path in generated_files:
                    file_size = file_path.stat().st_size / 1024  # KB
                    print(f"  - {file_path} ({file_size:.1f} KB)")
            else:
                print_warning("No compiled extensions found")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"Compilation failed with exit code {e.returncode}")
            if not self.verbose and e.stdout:
                print("STDOUT:", e.stdout)
            if not self.verbose and e.stderr:
                print("STDERR:", e.stderr)
            return False
        
        finally:
            # Clean up generated setup file
            Path(setup_file).unlink(missing_ok=True)

class TestRunner:
    """Run tests on compiled modules."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def test_import(self) -> bool:
        """Test if compiled modules can be imported."""
        print_step(4, "Testing module import")
        
        try:
            import risk_forecasting_core
            print_success("risk_forecasting_core imported successfully")
            
            # Test basic functionality
            if hasattr(risk_forecasting_core, 'TVPEVTCore'):
                print_success("TVPEVTCore class available")
            
            if hasattr(risk_forecasting_core, 'QuantileRegressionCore'):
                print_success("QuantileRegressionCore class available")
            
            return True
            
        except ImportError as e:
            print_error(f"Import failed: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality of compiled modules."""
        print_step(5, "Testing basic functionality")
        
        try:
            import numpy as np
            import risk_forecasting_core
            
            # Test TVP-EVT
            print("Testing TVP-EVT core...")
            tvp_core = risk_forecasting_core.TVPEVTCore()
            
            # Generate test data
            np.random.seed(42)
            test_returns = np.random.normal(0, 0.02, 500).astype(np.float64)
            
            # Test fitting
            tvp_core.fit(test_returns)
            print_success("TVP-EVT fitting successful")
            
            # Test forecasting
            forecast = tvp_core.forecast(horizon=1, confidence_levels=[0.95, 0.99])
            if isinstance(forecast, dict) and 'var_95' in forecast:
                print_success("TVP-EVT forecasting successful")
            
            # Test Quantile Regression
            print("Testing Quantile Regression core...")
            qr_core = risk_forecasting_core.QuantileRegressionCore()
            
            # Create test features
            n_features = 10
            X = np.random.randn(len(test_returns), n_features).astype(np.float64)
            
            # Test fitting
            beta = qr_core.fit_quantile(X, test_returns, 0.05)
            if len(beta) == n_features + 1:  # +1 for intercept
                print_success("Quantile Regression fitting successful")
            
            return True
            
        except Exception as e:
            print_error(f"Functionality test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

class BenchmarkRunner:
    """Run performance benchmarks."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def run_benchmarks(self) -> Dict:
        """Run comprehensive performance benchmarks."""
        print_step(6, "Running performance benchmarks")
        
        try:
            import numpy as np
            import time
            import risk_forecasting_core
            
            # Test data
            np.random.seed(42)
            sizes = [500, 1000, 2000]
            results = {}
            
            for size in sizes:
                print(f"\nBenchmarking with {size} observations...")
                
                test_returns = np.random.normal(0, 0.02, size).astype(np.float64)
                test_returns[::50] *= 3  # Add extreme events
                
                # Benchmark TVP-EVT
                tvp_core = risk_forecasting_core.TVPEVTCore()
                
                start_time = time.time()
                tvp_core.fit(test_returns)
                forecast = tvp_core.forecast()
                tvp_time = time.time() - start_time
                
                # Benchmark Quantile Regression
                qr_core = risk_forecasting_core.QuantileRegressionCore()
                X = np.random.randn(size, 15).astype(np.float64)
                
                start_time = time.time()
                beta = qr_core.fit_quantile(X, test_returns, 0.05)
                qr_time = time.time() - start_time
                
                results[size] = {
                    'tvp_evt_time': tvp_time,
                    'quantile_regression_time': qr_time,
                    'total_time': tvp_time + qr_time
                }
                
                print(f"  TVP-EVT: {tvp_time:.3f}s")
                print(f"  Quantile Regression: {qr_time:.3f}s")
                print(f"  Total: {tvp_time + qr_time:.3f}s")
            
            # Performance summary
            print("\n" + "="*50)
            print("PERFORMANCE SUMMARY")
            print("="*50)
            
            for size, timing in results.items():
                throughput = size / timing['total_time']
                print(f"Size {size}: {throughput:.0f} obs/sec")
            
            return results
            
        except Exception as e:
            print_error(f"Benchmark failed: {e}")
            return {}

def main():
    """Main build process."""
    parser = argparse.ArgumentParser(
        description="Build script for risk forecasting Cython modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                    # Standard build
  python build.py --clean --test     # Clean build with tests
  python build.py --benchmark        # Build and benchmark
  python build.py --verbose --parallel # Verbose parallel build
        """
    )
    
    parser.add_argument('--clean', action='store_true',
                       help='Clean build artifacts before building')
    parser.add_argument('--test', action='store_true',
                       help='Run tests after building')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel compilation')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling support')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install missing dependencies')
    parser.add_argument('--config-info', action='store_true',
                       help='Show build configuration and exit')
    
    args = parser.parse_args()
    
    # Initialize components
    config = BuildConfiguration()
    dependency_checker = DependencyChecker()
    builder = CythonBuilder(config, args.verbose)
    test_runner = TestRunner(args.verbose)
    benchmark_runner = BenchmarkRunner(args.verbose)
    
    # Show configuration info and exit if requested
    if args.config_info:
        print_header("BUILD CONFIGURATION")
        print(f"Platform: {config.platform}")
        print(f"Python: {config.python_version}")
        print(f"Architecture: {'64-bit' if config.is_64bit else '32-bit'}")
        print(f"Compiler: {config.compiler}")
        print(f"OpenMP: {'Yes' if config.enable_openmp else 'No'}")
        print(f"Compile args: {config.get_compile_args()}")
        return
    
    # Start build process
    print_header("RISK FORECASTING CYTHON BUILD")
    print(f"Platform: {config.platform} | Python: {config.python_version} | Compiler: {config.compiler}")
    
    start_time = time.time()
    success = True
    
    try:
        # Check dependencies
        if not dependency_checker.check_and_install_dependencies(args.install_deps):
            success = False
            return
        
        # Clean if requested
        if args.clean:
            builder.clean_build_artifacts()
        
        # Compile modules
        if not builder.compile_modules(args.parallel, args.profile):
            success = False
            return
        
        # Run tests if requested
        if args.test or args.benchmark:
            if not test_runner.test_import():
                success = False
                return
            
            if not test_runner.test_basic_functionality():
                success = False
                return
        
        # Run benchmarks if requested
        if args.benchmark:
            benchmark_results = benchmark_runner.run_benchmarks()
            if not benchmark_results:
                print_warning("Benchmarks failed or returned no results")
        
        total_time = time.time() - start_time
        
        print_header("BUILD COMPLETE")
        print_success(f"Total build time: {total_time:.2f}s")
        print_success("Risk forecasting Cython modules ready for use!")
        
        print("\nNext steps:")
        print("  from cython_wrapper_classes import OptimizedModelFactory")
        print("  model = OptimizedModelFactory.create_tvp_evt_model()")
        
    except KeyboardInterrupt:
        print_error("\nBuild interrupted by user")
        success = False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        success = False
    
    finally:
        if not success:
            print_header("BUILD FAILED")
            sys.exit(1)

if __name__ == "__main__":
    main()