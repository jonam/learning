from pylatexenc.latex2text import LatexNodes2Text
import re

with open('data/input.txt', 'r') as ff:
  data = ff.read()

converted_text = re.sub(r'\\\((.*?)\\\)', r'\\begin{equation}\1\\end{equation}', data)

# Convert the latex text to unicode using LatexNodes2Text
unicode_text = LatexNodes2Text().latex_to_text(converted_text)

header = "\documentclass[a4paper]{article}\n\\begin{document}\n"
footer = "\end{document}"
with open('data/output.tex', 'w') as ff:
  ff.write(header)
  ff.write(converted_text)
  ff.write(footer)

