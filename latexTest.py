# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:03:56 2016

@author: Tao
"""

min_latex = (r"\documentclass{article}"
             r"\begin{document}"
             r"Hello, world!"
             r"\end{document}")

from latex import build_pdf

# this builds a pdf-file inside a temporary directory
pdf = build_pdf(min_latex)

# look at the first few bytes of the header
print bytes(pdf)[:10]