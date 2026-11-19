// gpuping: the smallest command streams, one step at a time. Step 1: four CP_NOPs and a fence. Step 2: one
// CP_MEM_WRITE into a buffer of its own, read back by the CPU. The kernel has no gpu_recover: a hang stays a hang
// until a reboot, so each step is its own run and nothing here is a dispatch or a draw.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hakios-metal/gpu.hpp"
#include "hakios-metal/pm4.hpp"

using namespace hakios::metal;

int main(int argc, char** argv) {
    const int step = argc > 1 ? std::atoi(argv[1]) : 1;
    if (step != 1 && step != 2) {
        std::printf("gpuping: usage: gpuping 1|2\n");
        return 64;
    }
    std::string why;
    std::unique_ptr<Transport> t = openKernelGpu(&why);
    if (!t) {
        std::printf("gpuping: no kernel GPU: %s\n", why.c_str());
        return 2;
    }
    Gpu gpu(std::move(t));
    const uint64_t faults0 = gpu.faults();
    int err = 0;
    std::unique_ptr<Bo> cmd = gpu.newBo(4096, kBoWriteCombined, &err);
    std::unique_ptr<Bo> dst = gpu.newBo(4096, kBoWriteCombined, &err);
    if (!cmd || !dst) {
        std::printf("gpuping: FAIL GEM_NEW errno %d\n", err);
        return 3;
    }
    volatile uint32_t* d = static_cast<volatile uint32_t*>(dst->cpu());
    d[0] = 0xdeadbeefu;

    std::vector<uint32_t> w;
    for (int i = 0; i < 4; i++) w.push_back(pm4::type7(pm4::kNop, 0));
    std::vector<BoUse> uses;
    if (step == 2) {
        const uint64_t a = dst->gpuAddress();
        w.push_back(pm4::type7(pm4::kMemWrite, 3));
        w.push_back(uint32_t(a));
        w.push_back(uint32_t(a >> 32));
        w.push_back(0x610a610au);
        w.push_back(pm4::type7(pm4::kWaitMemWrites, 0));
        w.push_back(pm4::type7(pm4::kWaitForIdle, 0));
        uses.push_back(BoUse{dst.get(), true});
    }
    volatile uint32_t* c = static_cast<volatile uint32_t*>(cmd->cpu());
    for (size_t i = 0; i < w.size(); i++) c[i] = w[i];

    std::printf("gpuping: step %d: %zu words at %#llx\n", step, w.size(), (unsigned long long)cmd->gpuAddress());
    std::fflush(stdout);
    uint32_t fence = 0;
    int e = gpu.submit(0, *cmd, 0, uint32_t(w.size()), uses, &fence);
    std::printf("gpuping: submit -> %d, fence %u\n", e, fence);
    std::fflush(stdout);
    if (e != 0) return 4;

    const auto t0 = std::chrono::steady_clock::now();
    e = gpu.wait(0, fence, 2000000000ll);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::printf("gpuping: wait -> %d (%s) after %lld ms; FAULTS %llu -> %llu\n", e, e == 0 ? "done" : "NOT done",
                (long long)ms, (unsigned long long)faults0, (unsigned long long)gpu.faults());
    std::fflush(stdout);
    if (e != 0) return 5;

    int bad = 0;
    if (step == 2) {
        const int p = gpu.cpuPrep(*dst, 1, 2000000000ll);
        const uint32_t v = d[0];
        const bool ok = v == 0x610a610au;
        if (!ok) bad++;
        std::printf("gpuping: %s read back %#x (want 0x610a610a), cpuPrep -> %d\n", ok ? "ok  " : "FAIL", v, p);
        gpu.cpuFini(*dst);
    } else {
        std::printf("gpuping: ok   the GPU ran four CP_NOPs and signalled the fence\n");
    }
    return bad;
}
