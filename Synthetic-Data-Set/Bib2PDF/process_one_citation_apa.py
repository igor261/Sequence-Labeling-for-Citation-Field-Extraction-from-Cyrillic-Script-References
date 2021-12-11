import os
import subprocess
from subprocess import Popen, PIPE
from pdflatex import PDFLaTeX
import argparse

def create_parser():
    parser = argparse.ArgumentParser('')
    parser.add_argument("--filename", '-f', help = "big bibtex filename", required=True)
    parser.add_argument("--path", '-p', help = "path of bibtex and output", required=True)
    return parser.parse_args()

def create_pdf(filename):
   
    tex = filename + '.tex'
 
    with open(tex ,'w') as file:
        file.write('% !TeX program = lualatex\n')
        file.write('\\documentclass{article}\n')
        file.write('\\usepackage[T1,T2A]{fontenc}\n')
        file.write('\\usepackage[utf8]{inputenc}\n')
        file.write('\\usepackage[russian]{babel}\n')
        file.write('\\usepackage[numbers]{natbib}\n')
        file.write('\\begin{document}\n')
#         file.write('\\selectlanguage{russian}')
        file.write('\t\\nocite{*}\n')
        file.write(f'\t\\bibliography{{{filename}}}\n')
        file.write('\t\\bibliographystyle{apa.bst}\n')
        file.write('\\end{document}\n')
    
    x = subprocess.call(f'lualatex {tex}', shell=True)
    x = subprocess.call(f'bibtex {filename}', shell=True)
    p = Popen([f'lualatex {tex}'], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    stdout_data = p.communicate(input='R'.encode())[0]
    x = subprocess.call(f'pdflatex -interaction=nonstopmode {tex}', shell=True)
    if x != 0:
        print('Exit-code not 0, check result!')
        pass
        
def remove_files(file_path):
    os.remove(file_path+'.aux')
    os.remove(file_path+'.log')
    os.remove(file_path+'.blg')
    #os.remove(file_path) # remove bib file
    os.remove(file_path+'.bbl')
    os.remove(file_path+'.tex')

def main():
    args = create_parser()
    os.chdir(args.path)
    filename = args.filename
    with open(filename , encoding = 'utf-8') as big_bib:
        split_bib_bib = big_bib.read().split('@')[1:]
    for i in range(len(split_bib_bib)):
        if split_bib_bib[i].find("language= {russian}")==-1:
          bibtex = "@" + split_bib_bib[i]
  
          small_bib = "one_citation_" + str(i) + ".bib"
          with open(small_bib, 'w', encoding = 'utf-8') as bibfile:
              bibfile.write(bibtex)
  
          create_pdf(small_bib)
  
          path = os.getcwd() +'/'+ small_bib
          remove_files(path)
        else:
          pass
if __name__ == "__main__":        
    main()
    print('FERTIG')       

