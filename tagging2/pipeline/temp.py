import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR → tagging2/

output_path = os.path.join(BASE_DIR, "data", "tagged", "tagged_output.json")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("test")