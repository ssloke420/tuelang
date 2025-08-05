import os
import sys
import urllib.request
import shutil
import subprocess
import platform
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/ssloke420/tuelang/raw/main/src/interpreter.py"
INTERPRETER_FILE = "interpreter.py"

def get_install_config():
    """Get platform-specific installation configuration."""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "install_dir": "C:\\tuelang",
            "executable": "tuelang.bat",
            "executable_content": f'@echo off\npython "{Path("C:/tuelang") / INTERPRETER_FILE}" %*\n',
            "path_sep": ";",
            "shell_profile": None
        }
    elif system == "darwin":  # macOS
        home = Path.home()
        return {
            "install_dir": str(home / ".local" / "tuelang"),
            "executable": "tuelang",
            "executable_content": f'#!/bin/bash\npython3 "{home / ".local" / "tuelang" / INTERPRETER_FILE}" "$@"\n',
            "path_sep": ":",
            "shell_profile": str(home / ".zshrc")  # or .bash_profile
        }
    else:  # Linux
        home = Path.home()
        return {
            "install_dir": str(home / ".local" / "tuelang"),
            "executable": "tuelang",
            "executable_content": f'#!/bin/bash\npython3 "{home / ".local" / "tuelang" / INTERPRETER_FILE}" "$@"\n',
            "path_sep": ":",
            "shell_profile": str(home / ".bashrc")
        }

def install_cross_platform():
    """Cross-platform installation function."""
    config = get_install_config()
    system = platform.system()
    
    print(f"Installing TueLang on {system}...")
    
    # Create install directory
    Path(config["install_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Download interpreter
    print("Downloading interpreter...")
    req = urllib.request.Request(REPO_URL, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; TueLang-Installer/1.0)'
    })
    
    with urllib.request.urlopen(req) as response:
        with open(Path(config["install_dir"]) / INTERPRETER_FILE, 'wb') as f:
            shutil.copyfileobj(response, f)
    
    # Create executable wrapper
    executable_path = Path(config["install_dir"]) / config["executable"]
    with open(executable_path, 'w') as f:
        f.write(config["executable_content"])
    
    # Make executable on Unix systems
    if system != "Windows":
        os.chmod(executable_path, 0o755)
    
    # Add to PATH
    if system == "Windows":
        # Windows PATH handling (same as before)
        add_to_windows_path(config["install_dir"])
    else:
        # Unix PATH handling
        add_to_unix_path(config["install_dir"], config["shell_profile"])
    
    print(f"✓ TueLang installed to {config['install_dir']}")
    print(f"✓ Use 'tuelang <filename>' to run TueLang programs")

def add_to_windows_path(install_dir):
    """Add to Windows PATH (implementation from previous version)."""
    # Implementation from the previous installer
    pass

def add_to_unix_path(install_dir, shell_profile):
    """Add to Unix PATH via shell profile."""
    if not shell_profile or not os.path.exists(shell_profile):
        print(f"⚠ Could not find shell profile. Manually add to PATH:")
        print(f"export PATH=\"$PATH:{install_dir}\"")
        return
    
    path_line = f'export PATH="$PATH:{install_dir}"\n'
    
    # Check if already in profile
    with open(shell_profile, 'r') as f:
        if install_dir in f.read():
            print(f"✓ Already in PATH via {shell_profile}")
            return
    
    # Add to profile
    with open(shell_profile, 'a') as f:
        f.write(f'\n# TueLang\n{path_line}')
    
    print(f"✓ Added to PATH via {shell_profile}")
    print("⚠ Please restart your terminal or run: source ~/.bashrc")

if __name__ == "__main__":
    install_cross_platform()
