import argparse
import tempfile
from pylatexenc.latex2text import LatexNodes2Text
import re
import os

# Create the parser
parser = argparse.ArgumentParser(description='Convert ChatGPT to PDF.')
# Add the chatgptFile argument
parser.add_argument('--chatgptFile', required=True, help='Output file path')

# Parse the arguments
args = parser.parse_args()
# Print the chatgptFile argument value
print(f"Output file is {args.chatgptFile}")

# Creating a temporary file in TMP directory
with tempfile.NamedTemporaryFile(dir='.', delete=False) as tmpfile:
    print(f"Temporary file created at: {tmpfile.name}")
    # You can use tmpfile.name as the path to the temporary file
    # Remember, the file will be deleted if you close it, due to delete=False

with open(args.chatgptFile, 'r') as ff:
  data = ff.read()

converted_text = re.sub(r'\\\((.*?)\\\)', r'\\begin{equation}\1\\end{equation}', data)

# Convert the latex text to unicode using LatexNodes2Text
unicode_text = LatexNodes2Text().latex_to_text(converted_text)

header = "\documentclass[a4paper]{article}\n\\begin{document}\n"
footer = "\end{document}"
with open(tmpfile.name, 'w') as ff:
  ff.write(header)
  ff.write(converted_text)
  ff.write(footer)

os.system(f"pdflatex {tmpfile.name}")
os.system(f"mv {tmpfile.name}.pdf ./{args.chatgptFile}.pdf")
os.system(f"rm {tmpfile.name}")
os.system(f"rm {tmpfile.name}.aux")
os.system(f"rm {tmpfile.name}.log")
