import re

with open('keyword/output.txt', 'r',encoding="utf8") as f:
    lines = f.readlines()

with open('Keyword/specifications.txt', 'w',encoding="utf8") as f:
    for line in lines:
        # This regex pattern matches any line that contains only alphabets (case insensitive)
        if re.match('^[a-zA-Z]*$', line):
            f.write(line)
