// gpuinfo: a read-only look at the kernel's GPU driver, for the A610 profile. Opens the service, reads GET_PARAM,
// makes one 4 KiB buffer and gives it back. No command stream is submitted, so nothing here can hang the GPU.
#include <cstdio>
#include <string>

#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/gpu.hpp"

using namespace hakios::metal;

int main() {
    int bad = 0;
    std::string why;
    std::unique_ptr<Transport> t = openKernelGpu(&why);
    if (!t) {
        std::printf("gpuinfo: no kernel GPU: %s\n", why.c_str());
        return 2;
    }
    Gpu gpu(std::move(t));

    struct Item { const char* name; uint32_t id; bool check; uint64_t want; };
    const Item items[] = {
        {"GPU_ID", kParamGpuId, true, a6xx::kGpuId},
        {"GMEM_SIZE", kParamGmemSize, true, a6xx::kGmemBytes},
        {"MAX_FREQ", kParamMaxFreq, false, 0},
        {"GMEM_BASE", kParamGmemBase, false, 0},
        {"PRIORITIES", kParamPriorities, false, 0},
        {"FAULTS", kParamFaults, true, 0},
        {"HIGHEST_BANK_BIT", kParamHighestBankBit, true, 13},
        {"UBWC_SWIZZLE", kParamUbwcSwizzle, true, 7},
        {"MACROTILE_MODE", kParamMacrotileMode, true, 0},
    };
    for (const Item& it : items) {
        uint64_t v = 0;
        int e = gpu.param(it.id, &v);
        if (e != 0) {
            std::printf("gpuinfo: FAIL %-17s errno %d\n", it.name, e);
            bad++;
        } else if (!it.check) {
            std::printf("gpuinfo: --   %-17s %#llx\n", it.name, (unsigned long long)v);
        } else {
            const bool ok = v == it.want;
            if (!ok) bad++;
            std::printf("gpuinfo: %-4s %-17s %#llx (a6xx.hpp: %#llx)\n", ok ? "ok" : "FAIL", it.name,
                        (unsigned long long)v, (unsigned long long)it.want);
        }
    }
    // the chip id: a6xx.hpp's kChipId says 0x06010000, and the kernel adds the low byte (0x06010001)
    const uint32_t chip = gpu.chipId();
    const bool known = (chip & ~0xffu) == (a6xx::kChipId & ~0xffu);
    if (!known) bad++;
    std::printf("gpuinfo: %-4s %-17s %#x (a6xx.hpp: %#x)\n", known ? "ok" : "FAIL", "CHIP_ID", chip, a6xx::kChipId);

    int err = 0;
    std::unique_ptr<Bo> bo = gpu.newBo(4096, kBoWriteCombined, &err);
    if (!bo) {
        std::printf("gpuinfo: FAIL GEM_NEW errno %d\n", err);
        bad++;
    } else {
        std::printf("gpuinfo: ok   GEM_NEW           handle %u size %llu iova %#llx cpu %s\n", bo->handle(),
                    (unsigned long long)bo->size(), (unsigned long long)bo->gpuAddress(),
                    bo->cpu() != nullptr ? "mapped" : "not mapped");
    }
    bo.reset();
    std::printf("gpuinfo: %d failures\n", bad);
    return bad;
}
