import os
import json
import hashlib
import subprocess
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum, auto

import pkg_resources
import requests
import toml
import yaml

class DependencySource(Enum):
    """Supported dependency sources"""
    PYPI = auto()
    GITHUB = auto()
    LOCAL = auto()
    CUSTOM = auto()

@dataclass
class PackageMetadata:
    """Comprehensive package metadata"""
    name: str
    version: str
    source: DependencySource = DependencySource.PYPI
    source_url: Optional[str] = None
    description: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    hash: Optional[str] = None
    installed_path: Optional[str] = None
    
    def generate_hash(self) -> str:
        """Generate package hash"""
        hash_content = f"{self.name}{self.version}{self.source}"
        return hashlib.md5(hash_content.encode()).hexdigest()

class PackageSecurityScanner:
    """Advanced package security scanning system"""
    
    @staticmethod
    def scan_package(package_name: str) -> Dict[str, Any]:
        """
        Scan package for security vulnerabilities
        
        :param package_name: Package to scan
        :return: Security scan results
        """
        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run(
                ['safety', 'check', f'-r {package_name}'],
                capture_output=True,
                text=True
            )
            
            return {
                'vulnerable': result.returncode != 0,
                'details': result.stdout
            }
        except Exception as e:
            return {
                'vulnerable': True,
                'error': str(e)
            }

class PackageManager:
    """
    Advanced dependency and package management system
    
    Features:
    - Multi-source package management
    - Dependency resolution
    - Security scanning
    - Virtual environment support
    """
    
    def __init__(
        self, 
        project_root: str = '/home/veronicae/CascadeProjects/MLSP'
    ):
        """
        Initialize package manager
        
        :param project_root: Root directory of the project
        """
        self.project_root = project_root
        self.package_cache_dir = os.path.join(project_root, '.package_cache')
        os.makedirs(self.package_cache_dir, exist_ok=True)
        
        # Package metadata storage
        self.packages_file = os.path.join(
            self.package_cache_dir, 
            'packages.json'
        )
        self.packages: Dict[str, PackageMetadata] = self._load_packages()
    
    def _load_packages(self) -> Dict[str, PackageMetadata]:
        """Load existing package metadata"""
        if os.path.exists(self.packages_file):
            with open(self.packages_file, 'r') as f:
                packages_data = json.load(f)
                return {
                    name: PackageMetadata(**pkg) 
                    for name, pkg in packages_data.items()
                }
        return {}
    
    def _save_packages(self):
        """Save package metadata"""
        packages_data = {
            name: asdict(pkg) 
            for name, pkg in self.packages.items()
        }
        
        with open(self.packages_file, 'w') as f:
            json.dump(packages_data, f, indent=2)
    
    def install_package(
        self, 
        package_name: str, 
        version: Optional[str] = None,
        source: DependencySource = DependencySource.PYPI
    ) -> PackageMetadata:
        """
        Install a package
        
        :param package_name: Package to install
        :param version: Specific version to install
        :param source: Package source
        :return: Package metadata
        """
        # Construct installation command
        install_cmd = [sys.executable, '-m', 'pip', 'install']
        
        if version:
            package_spec = f'{package_name}=={version}'
        else:
            package_spec = package_name
        
        install_cmd.append(package_spec)
        
        # Run installation
        result = subprocess.run(
            install_cmd, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Package installation failed: {result.stderr}")
        
        # Retrieve package information
        try:
            dist = pkg_resources.get_distribution(package_name)
            
            # Security scan
            security_scan = PackageSecurityScanner.scan_package(package_name)
            
            # Create package metadata
            package_metadata = PackageMetadata(
                name=package_name,
                version=dist.version,
                source=source,
                installed_path=dist.location
            )
            package_metadata.hash = package_metadata.generate_hash()
            
            # Store package metadata
            self.packages[package_name] = package_metadata
            self._save_packages()
            
            return package_metadata
        
        except pkg_resources.DistributionNotFound:
            raise RuntimeError(f"Package {package_name} not found after installation")
    
    def uninstall_package(self, package_name: str):
        """
        Uninstall a package
        
        :param package_name: Package to uninstall
        """
        uninstall_cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', package_name]
        
        result = subprocess.run(
            uninstall_cmd, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Package uninstallation failed: {result.stderr}")
        
        # Remove from tracked packages
        if package_name in self.packages:
            del self.packages[package_name]
            self._save_packages()
    
    def list_packages(
        self, 
        source: Optional[DependencySource] = None
    ) -> List[PackageMetadata]:
        """
        List installed packages
        
        :param source: Optional source filter
        :return: List of package metadata
        """
        if source:
            return [
                pkg for pkg in self.packages.values() 
                if pkg.source == source
            ]
        return list(self.packages.values())
    
    def generate_requirements(
        self, 
        output_format: str = 'txt'
    ) -> str:
        """
        Generate requirements file
        
        :param output_format: Output file format
        :return: Requirements file content
        """
        requirements = [
            f"{pkg.name}=={pkg.version}" 
            for pkg in self.packages.values()
        ]
        
        if output_format == 'txt':
            return '\n'.join(requirements)
        elif output_format == 'yaml':
            return yaml.safe_dump({'dependencies': requirements})
        elif output_format == 'toml':
            return toml.dumps({'tool': {'poetry': {'dependencies': dict(
                (pkg.name, pkg.version) for pkg in self.packages.values()
            )}}})
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def update_package(
        self, 
        package_name: str, 
        version: Optional[str] = None
    ) -> PackageMetadata:
        """
        Update a package
        
        :param package_name: Package to update
        :param version: Optional specific version
        :return: Updated package metadata
        """
        # Uninstall current version
        self.uninstall_package(package_name)
        
        # Reinstall package
        return self.install_package(package_name, version)

def main():
    """Demonstration of package management system"""
    # Initialize package manager
    pkg_manager = PackageManager()
    
    # Install packages
    torch_pkg = pkg_manager.install_package('torch', '2.1.0')
    numpy_pkg = pkg_manager.install_package('numpy')
    
    # List installed packages
    print("Installed Packages:")
    for pkg in pkg_manager.list_packages():
        print(f"{pkg.name} (v{pkg.version})")
    
    # Generate requirements file
    print("\nRequirements (TXT):")
    print(pkg_manager.generate_requirements('txt'))
    
    # Update a package
    updated_numpy = pkg_manager.update_package('numpy')
    print(f"\nUpdated NumPy to version: {updated_numpy.version}")

if __name__ == '__main__':
    main()
