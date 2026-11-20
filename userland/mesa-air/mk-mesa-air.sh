#!/bin/bash
# Mesa 25.0.7 for hakios-metal's compiler (AIR -> NIR -> ir3), cross-built for iOS 16 arm64 with Xcode's clang.
# A tree of its own, apart from userland/mesa (the GL library that draws the screen): only static archives are built.
set -euo pipefail
VERSION=25.0.7
SHA256=592272df3cf01e85e7db300c449df5061092574d099da275d19e97ef0510f8a6
ROOT=$(cd "$(dirname "$0")" && pwd)
B=$ROOT/build
mkdir -p "$B"
SDK=$(xcrun --sdk iphoneos --show-sdk-path)
CC=$(xcrun --sdk iphoneos -f clang)
cd "$B"
PATCHES=$( (cat "$ROOT"/patches/*.patch; find "$ROOT/air" "$ROOT/hakios-drm" -type f | sort | xargs cat) | shasum -a 256 | cut -d' ' -f1)
if [ ! -d mesa-$VERSION ] || [ "$(cat mesa-$VERSION/.air-patches 2>/dev/null)" != "$PATCHES" ]; then
	[ -f mesa-$VERSION.tar.xz ] || curl -sL --fail -o mesa-$VERSION.tar.xz https://archive.mesa3d.org/mesa-$VERSION.tar.xz
	echo "$SHA256  mesa-$VERSION.tar.xz" | shasum -a 256 -c --quiet
	rm -rf mesa-$VERSION build-*
	tar xJf mesa-$VERSION.tar.xz
	for p in "$ROOT"/patches/*.patch; do patch -s --no-backup-if-mismatch -d mesa-$VERSION -p1 < "$p"; done
	echo "$PATCHES" > mesa-$VERSION/.air-patches
fi
rm -rf mesa-$VERSION/src/hakios-drm mesa-$VERSION/src/compiler/air
cp -r "$ROOT/hakios-drm" mesa-$VERSION/src/hakios-drm
cp -r "$ROOT/air" mesa-$VERSION/src/compiler/air
if [ ! -x venv/bin/python3 ]; then
	python3 -m venv venv
	venv/bin/pip -q install mako pyyaml packaging
fi
export PATH=$B/venv/bin:/opt/homebrew/opt/bison/bin:$PATH
cat > cross.ini <<INI
[binaries]
c = ['$CC', '-target', 'arm64-apple-ios16.0', '-isysroot', '$SDK']
cpp = ['$CC++', '-target', 'arm64-apple-ios16.0', '-isysroot', '$SDK', '-fuse-cxa-atexit']
objc = ['$CC', '-target', 'arm64-apple-ios16.0', '-isysroot', '$SDK']
ar = '$(xcrun -f ar)'
ranlib = '$(xcrun -f ranlib)'
strip = '$(xcrun -f strip)'
pkg-config = 'false'

[host_machine]
system = 'darwin'
subsystem = 'ios'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'

[properties]
needs_exe_wrapper = true
INI
BUILD=build-softpipe-freedreno
[ -f $BUILD/build.ninja ] || meson setup $BUILD mesa-$VERSION --cross-file cross.ini --wrap-mode=nofallback \
	--prefix=/redwood --libdir=lib --buildtype=release -Db_ndebug=true \
	-Dplatforms= -Dgallium-drivers=softpipe,freedreno -Dfreedreno-kmds=msm -Dvulkan-drivers= -Dglx=disabled -Degl=disabled -Dgbm=disabled \
	-Dosmesa=true -Dllvm=disabled -Dshared-glapi=enabled -Dgles1=disabled -Dgles2=enabled -Dopengl=true \
	-Dzstd=disabled -Dxmlconfig=disabled -Dexpat=disabled -Dshader-cache=disabled -Dvalgrind=disabled \
	-Dlibunwind=disabled -Dglvnd=disabled -Dvideo-codecs= -Dtools= -Dgallium-vdpau=disabled -Dgallium-va=disabled \
	-Dgallium-xa=disabled -Dlmsensors=disabled -Dbuild-tests=false > meson-setup.log
ninja -C $BUILD -t targets all > targets.txt
for t in src/compiler/air/libair.a src/freedreno/ir3/libfreedreno_ir3.a src/freedreno/isa/libir3encode.a \
         src/freedreno/isa/libir3decode.a src/compiler/isaspec/libisaspec.a src/compiler/nir/libnir.a \
         src/compiler/libcompiler.a src/freedreno/fdl/libfreedreno_layout.a src/freedreno/common/libfreedreno_common.a \
         src/util/libmesa_util.a src/util/libmesa_util_sse41.a src/util/blake3/libblake3.a src/c11/impl/libmesa_util_c11.a; do
	if grep -q "^$t:" targets.txt; then
		ninja -C $BUILD "$t" >> ninja.log 2>&1 || { echo "GAGAL: $t"; tail -n 60 ninja.log; exit 1; }
		echo "ok   $t"
	else
		echo "(tidak ada target: $t)"
	fi
done
{
	echo "== archives"; ls -l $BUILD/src/compiler/air/libair.a $BUILD/src/freedreno/ir3/libfreedreno_ir3.a 2>&1
	echo "== all .a files"; find $BUILD -name "*.a" | sort
	echo "== compile line of ir3.c.o"
	grep -m1 -A4 '^build src/freedreno/ir3/libfreedreno_ir3.a.p/ir3.c.o:' $BUILD/build.ninja
} > report.txt
cat report.txt | cut -c1-300 | head -60
