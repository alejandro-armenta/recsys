import random
import json

from absl import flags
from absl import app

from typing import Dict

import pin_util

FLAGS = flags.FLAGS
_INPUT_FILE=flags.DEFINE_string('input_file',None,'Input cat json file.')
_OUTPUT_HTML=flags.DEFINE_string('output_html',None,'The output html file.')
_NUM_ITEMS=flags.DEFINE_integer('num_items', 10, 'Number of items to recommend.')

def read_catalog(catalog: str) -> Dict[str, str]:
    with open(catalog, 'r') as f:
        data = f.read()
        #print(data)

    result = json.loads(data)    

    print(type(result))

    return result

def dump_html(subset, output_html:str) -> None:

    with open(output_html, 'w') as f:
        f.write('<HTML>')
        f.write("""
<TABLE>
<tr><th>Key</th><th>Category</th><th>Image</th><tr>
""")
        
        for item in subset:

            key,category = item

            url = pin_util.key_to_url(key)
            
            #print(url)

            img_url = f"<img src={url}>"


            out = f"<tr><td>{key}</td><td>{category}</td><td>{img_url}</td></tr>\n"

            print(out)

            f.write(out)
        
        f.write('</TABLE>\n</HTML>')
        
    

def main(argv):
    #print(_INPUT_FILE.value)
    #print(_OUTPUT_HTML.value)

    catalog = read_catalog(_INPUT_FILE.value)

    catalog = list(catalog.items())
    #es una tuple de key value!
    #print(catalog[0])

    random.shuffle(catalog)

    #print(catalog[0])

    dump_html(catalog[:_NUM_ITEMS.value], _OUTPUT_HTML.value)
    #dump_html()


if __name__ == "__main__":
    app.run(main)

