import subprocess

subprocess.run(["mkdir", "-p", "dist"])

subprocess.run(["python3", "generate_model.py"])

subprocess.run(["streamlit", "run", "app.py"])