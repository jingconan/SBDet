find . -iname "__pycache__" -exec rm -r '{}' ';'                              
find . -iname "*.pyc" -exec rm '{}' ';'                                       
find . -iname "tags" -exec rm '{}' ';'         
