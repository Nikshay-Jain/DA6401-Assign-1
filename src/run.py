import subprocess

venv_path = r"C:\Users\niksh\Desktop\Assign 1\venv\Scripts\activate"
script_path = r"C:\Users\niksh\Desktop\Assign 1\src\train.py"

for i in range(50):
    print(f"Running iteration {i+1}/50...")
    subprocess.run(["powershell", "-Command", f"Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; & '{venv_path}'; python '{script_path}'"])
