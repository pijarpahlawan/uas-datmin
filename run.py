import subprocess
subprocess.check_call(["pip", "install", "pandas"])
subprocess.check_call(["pip", "install", "scikit-learn"])

subprocess.run(["mkdir", "-p", "dist"])

subprocess.run(["python3", "generate_model.py"])

subprocess.run(["streamlit", "run", "app.py"])