# Convert output of a chatgpt to PDF

## One time install

Install mactex, this takes a while.

```
brew install mactex
```

Then create a new Mac terminal (for the pdflatex command to be available).


## pip install

```
pip install pylatexenc
```

## Convert chatgpt to PDF

Copy the output to a clipboard and save that to a file, say chatgpt.txt

```
python3 chatgpt2pdf.py --chatgptFile chatgpt.txt
```
