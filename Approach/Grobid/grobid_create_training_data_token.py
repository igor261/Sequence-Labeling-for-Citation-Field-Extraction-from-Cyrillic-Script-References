
import pandas as pd
import os
import xml.etree.ElementTree as ET

import re
import numpy as np

def listdir_path(d):
    # Return full path of all files & directories in directory
    list_full_path = []
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        list_full_path.append(full_path)
    return list_full_path
    
# path = "/local/users/ulede/BERT/labelled_text"
path = ".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\Labeled_text_data"
apa = listdir_path(os.path.join(path,"apa_fine_grained_clean"))
gost2003 = listdir_path(os.path.join(path,"gost2003_fine_grained_clean"))
gost2008 = listdir_path(os.path.join(path,"gost2008_fine_grained_clean"))
gost2006 = listdir_path(os.path.join(path,"gost2006_fine_grained_clean"))
files = apa + gost2003 + gost2008 + gost2006
print("Anzahl files: " + str(len(files)))
label_dict_beg = {"title":"""<title level="a">""" ,
"booktitle":"""<title level="j">""",
"journal":"""<title level="j">""",
"publisher":"""<publisher>""",
"volume":"""<biblScope unit="volume">""",
"year":"""<date>""",
"address":"""<pubPlace>""",
"number":"""<biblScope unit="issue">""",
"pages":"""<biblScope unit="page">""",
"pagetotal":"""<biblScope unit="page">"""}

label_dict_end = {"title":"""</title>""" ,
"booktitle":"""</title>""",
"journal":"""</title>""",
"publisher":"""</publisher>""",
"volume":"""</biblScope>""",
"year":"""</date>""",
"address":"""</pubPlace>""",
"number":"""</biblScope>""",
"pages":"""</biblScope>""",
"pagetotal":"""</biblScope>"""}

# <author>,<publisher> tag is the same in GROBID training TEI
all_ref = []
file_ids = []
for f in files:
    file = open(f,"r",encoding="utf-8")
    ref = file.read().split("\n\n")
    for i in range(len(ref)):
        if "apa_" in f:
            file_id = "apa_"+f.split("/")[-1].replace(".xml","")
        elif "gost2003_" in f:
            file_id = "gost2003_"+f.split("/")[-1].replace(".xml","")
        elif "gost2006_" in f:
            file_id = "gost2006_"+f.split("/")[-1].replace(".xml","")
        elif "gost2008_" in f:
            file_id = "gost2008_"+f.split("/")[-1].replace(".xml","")
        file_ids.append(file_id)
    all_ref+=ref

tei_ref = []

LABELS_TXT = ["year","title","booktitle","volume","number","pages","address","publisher","pagetotal","journal"]

for ref in all_ref:
    text_ref=""
    ref_labels = []
    for l in LABELS_TXT:
        if f"<{l}>" in ref:
            ref_labels.append(l)
            
    for l in ref_labels:
        ref = ref.replace(f"<{l}>",label_dict_beg[l]).replace(f"</{l}>",label_dict_end[l])
        
    ref = ref.replace("<i/>","").replace("\x16","--").replace("< i / >","").replace("""< ""","""&gt """).replace(""" >""",""" &lt""") # Anpassungen bewerten..
    tei_ref.append(ref)
    
    

# path_output = "/local/users/ulede/grobid_citation_token_level"
path_output = ".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Grobid\grobid_training_data"

begin_file = """<?xml version="1.0" ?>
<TEI xml:space="preserve" xmlns="http://www.tei-c.org/ns/1.0" xmlns:xlink="http://www.w3.org/1999/xlink" 
 xmlns:mml="http://www.w3.org/1998/Math/MathML">
	<teiHeader>
		<fileDesc xml:id="0"/>
	</teiHeader>
	<text>
		<front/>
		<body/>
		<back>
<listBibl>"""
end_file = """		</listBibl>
	</back>
	</text>
</TEI>"""

for r,f in zip(tei_ref,file_ids):
    output_text = begin_file + "\n<bibl>" +r+"</bibl>\n" +end_file
    myfile = open(os.path.join(path_output, f+".references.tei.xml"), "w",encoding="utf-8")
    myfile.write(output_text)
    myfile.close()
print("DONE")
print(str(len(tei_ref))+ " files created!")

files_grobid = listdir_path(path_output)

errors = []
for f in files_grobid:
    try:
        tree = ET.parse(f)
        root = tree.getroot()
    except:
        
        errors.append(f)
        os.remove(f)
print("errors in:  " + str(len(errors)) + " files. Files removed!")