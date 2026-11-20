#!/bin/bash
set -euo pipefail
[[ "${MACHINE_CONFIG:-}" == SWEET ]] || exit 0
f=iokit/Kernel/arm/SweetFramebuffer.cpp
[[ -f $f ]] || { echo "0139: tidak ada $f" >&2; exit 1; }
python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if 'PE_parse_boot_argn("metal"' in s:
    print("0139: sudah ada, dilewati")
    sys.exit(0)
old = '\t\taccel->setName("sgx");\n'
if s.count(old) != 1:
    sys.exit('0139: setName("sgx") tidak cocok tepat sekali (%d)' % s.count(old))
new = old + (
    '\t\t/* boot-arg metal=N: the accelerator names the Metal plug-in (docs/metal-plan.md step 1 of hakios).\n'
    '\t\t * Without the argument nothing changes: a device that cannot draw yet breaks the window server. */\n'
    '\t\tuint32_t metalOn = 0;\n'
    '\t\tif (PE_parse_boot_argn("metal", &metalOn, sizeof(metalOn)) && metalOn != 0) {\n'
    '\t\t\taccel->setProperty("MetalPluginName", "HakiosMetal");\n'
    '\t\t\taccel->setProperty("MetalPluginClassName", "HakiosMetalDevice");\n'
    '\t\t\tIOLog("SWEET FB: metal=%u: the accelerator names HakiosMetal as its Metal plug-in\\n", metalOn);\n'
    '\t\t}\n')
open(p, "w").write(s.replace(old, new))
print("0139: diubah")
PY
