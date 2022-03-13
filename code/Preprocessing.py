
import json

import requests



with open(r'C:\Users\alexl_g8yj9pc\Documents\DAEN690\data\tagged_entities\HM_tags_0305.json', mode='r', encoding='UTF-8') as f:

        data = json.load(f)



for each in data:

    # You may have to change your url depending on where is hosted.

    url = f"http://localhost:8080{each['data'][list(each['data'].keys())[0]]}"

    # You can Find your token clicking at upper right panel button in Label-Studio and Account & Settings. 

    r = requests.get(url, headers={'Authorization':'Token 9f1204eb7f66aef3f372d6aef5170715e59a0b4e', })

    r.encoding = 'UTF-8'

    each['data'][list(each['data'].keys())[0]] = r.text



with open('HM_conll.json', mode='w', encoding = 'UTF-8' ) as f:

    json.dump(data, f)

import os
os.getcwd()






