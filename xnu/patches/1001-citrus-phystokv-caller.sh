#!/bin/bash
set -euo pipefail
[[ "${MACHINE_CONFIG:-}" == SWEET ]] || exit 0
f=osfmk/arm64/arm_vm_init.c
echo "1001: cwd=$PWD"
[[ -f $f ]] || { echo "1001: tidak ada $f" >&2; exit 1; }
grep -n "illegal PA" "$f" || true
python3 - "$f" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'panic\("%s: illegal PA:([^"]*)",(\s*)([^,()]+),',
               r'panic("%s [caller %p]: illegal PA:\1",\2\3, __builtin_return_address(0),', s)
print('1001: diubah', n, 'tempat')
if n < 2:
    sys.exit('1001: kurang dari 2 tempat cocok')
open(p, 'w').write(s)
EOF
