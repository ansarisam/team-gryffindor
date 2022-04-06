# %%
from tika import parser
import os
import re
import glob
import pandas as pd


import pytesseract
from pdf2image import convert_from_path

from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# %%
# Data file location 
dir = r'..\data\Credit_Card_Agreements_2021_Q3'
output_dir = r'..\data\all_OCR_results'

assigned_letters = ['M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    '1','2','3','4','5','6','7','8','9','0']

already_done = ['A','B','C','D','E','F', 'G','H','I','J','K',
                    'L']
for letter in assigned_letters:
    print(letter)
    file =[]
    #B subfolder you guys can change the pattern to find other folders
    pattern = re.compile('^'+letter)
    
    result = []
    ext = "*.pdf"
    
    for dirpath, dirnames, filenames in os.walk(dir):
        result = result + [dirname for dirname in dirnames if pattern.match(dirname)]
        for name in result:
            file += glob.glob(os.path.join(dirpath,name,ext))
            
        
    
    # %%
    df = pd.DataFrame(columns=('filename','text'))
    
    
    # %%
    for idx, filename in enumerate(file):
       data = parser.from_file(filename)
       text = data['content']
       if not text:
           print('Tika failed- using Tesseract')
           pages = convert_from_path(filename, 500)
           text = ' '
           for pageNum, imgBlob in enumerate(pages): 
               text += pytesseract.image_to_string(imgBlob,lang='eng')
               
       df.loc[idx] = [filename, text]
        
    
    # %%
    #Exporting to text file
    for index, row in df.iterrows():
        output_tokens = row['filename'].split('\\')
        new_filename = output_tokens[4]+'-'+output_tokens[5]
        output_path = os.path.join(output_dir,new_filename.replace('.pdf', '.txt'))
        with open(output_path, 'w', encoding='utf-8') as f:
            clean_text = str(row['text']).lstrip()
            f.write(clean_text)


# %%
