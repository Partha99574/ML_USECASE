import subprocess

def test_training_runs():
    result = subprocess.run(["python", "src/train.py"], capture_output=True)
    assert result.returncode == 0