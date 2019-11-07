import sys

buf = []
for line in sys.stdin:
    if line.startswith('#'):
        continue
    if '.' in line.split("\t")[0]:
        continue
    if '-' in line.split("\t")[0]:
        continue
    if not line.strip():
        if buf:
            print(" ".join(buf))
            buf = []
        else:
            continue
    else:
        buf.append(line.strip().split("\t")[1])
if buf:
    print(" ".join(buf))
