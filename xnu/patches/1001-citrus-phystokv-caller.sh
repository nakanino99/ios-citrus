#!/bin/bash
# Diagnostik sementara: pemanggil phystokv_range ikut tercetak di panic "illegal PA".
# dxb menjalankan skrip ini dari root source XNU, sebelum *.patch diterapkan.
set -euo pipefail
[[ "${MACHINE_CONFIG:-}" == SWEET ]] || exit 0
f=${1:-.}/osfmk/arm64/arm_vm_init.c
[[ -f $f ]] || { echo "tidak ada $f" >&2; exit 1; }
python3 - "$f" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
if '[caller %p]' in s:
    sys.exit(0)
k = 'panic("%s: illegal PA: '
i = s.find(k)
if i < 0:
    sys.exit('teks panic tidak ditemukan')
j = s.find('__func__,', i)
if j < 0 or j - i > 200:
    sys.exit('__func__ tidak ada dekat panic')
s = (s[:i] + 'panic("%s [caller %p]: illegal PA: ' + s[i+len(k):j]
     + '__func__, __builtin_return_address(0),' + s[j+len('__func__,'):])
open(p, 'w').write(s)
print('patched', p)
EOF
