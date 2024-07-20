import os
import sys
import urllib.request
import shutil
import winreg

# Configuration
REPO_URL = "https://github.com/yourusername/tuelang/raw/main/src/interpreter.py"
INSTALL_DIR = "C:\\tuelang"
INTERPRETER_FILE = "interpreter.py"

def download_interpreter():
    print(f"Downloading Tuelang interpreter from {REPO_URL}...")
    
    # Download the interpreter file
    response = urllib.request.urlopen(REPO_URL)
    with open(os.path.join(INSTALL_DIR, INTERPRETER_FILE), 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    print("Download complete.")

def add_to_path():
    print(f"Adding {INSTALL_DIR} to PATH...")

    # Access the registry to update the PATH
    try:
        reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_WRITE)
        current_path = winreg.QueryValueEx(reg_key, "PATH")[0]
        new_path = f"{current_path};{INSTALL_DIR}"
        winreg.SetValueEx(reg_key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
        winreg.CloseKey(reg_key)
        print("PATH updated successfully.")
    except Exception as e:
        print(f"Failed to update PATH: {e}")
        sys.exit(1)

def setup():
    # Create the installation directory if it does not exist
    if not os.path.exists(INSTALL_DIR):
        os.makedirs(INSTALL_DIR)

    download_interpreter()
    add_to_path()
    
    print(f"Tuelang has been installed to {INSTALL_DIR}")
    print("You may need to restart your command prompt or system for changes to take effect.")

if __name__ == "__main__":
    setup()
