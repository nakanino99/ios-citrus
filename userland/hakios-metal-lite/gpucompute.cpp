// gpucompute: the smallest compute dispatches, one run per variant. No gpu_recover in this kernel: a hang stays a
// hang until a reboot, so run the steps in order and read each result before the next.
//   1: end_only, 1 group, without the dispatch's own tail (nothing is written; the kernel's fence closes it)
//   2: end_only, 1 group, the whole sequence (tail on)
//   3: fill_id, 16 groups of 64, tail on, c = 0x1234: the first shader that WRITES; every word is checked
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "fixtures/end_only.hpp"
#include "fixtures/fill_id.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/compute.hpp"

using namespace hakios::metal;

static void onAlarm(int) {
    const char m[] = "gpucompute: no answer after 20 s: the GPU is probably hung (reboot to recover)\n";
    if (write(1, m, sizeof m - 1) < 0) {}
    _exit(7);
}

// fill_id's answer for one invocation (probe.cpp): gid * K + c, modulo 2^32
static uint32_t expectedWord(uint32_t gid, uint32_t c) { return gid * 0x9E3779B1u + c; }

// a value no invocation of the grid can write with this c, so a word still holding it was not written (probe.cpp)
static uint32_t canaryFor(uint32_t words, uint32_t c) {
    uint32_t canary = 0xa5a5a5a5u;
    for (;;) {
        bool clash = false;
        for (uint32_t gid = 0; gid < words && !clash; gid++) clash = expectedWord(gid, c) == canary;
        if (!clash) return canary;
        canary += 0x01010101u;
    }
}

int main(int argc, char** argv) {
    const int step = argc > 1 ? std::atoi(argv[1]) : 0;
    if (step < 1 || step > 3) {
        std::printf("gpucompute: usage: gpucompute 1|2|3   (1: end_only no tail, 2: end_only whole sequence, 3: fill_id)\n");
        return 64;
    }
    std::string why;
    std::unique_ptr<Transport> t = openKernelGpu(&why);
    if (!t) {
        std::printf("gpucompute: no GPU: %s\n", why.c_str());
        return 2;
    }
    Gpu gpu(std::move(t));
    const uint64_t faults0 = gpu.faults();
    auto queue = CommandQueue::make(gpu, &why);
    if (!queue) {
        std::printf("gpucompute: FAIL no queue: %s\n", why.c_str());
        return 3;
    }
    auto cb = queue->commandBuffer();
    compute::Options o;
    if (step == 1) o.tail = false;

    const uint32_t words = 1024, c = 0x00001234;
    std::shared_ptr<Buffer> out;
    uint32_t canary = 0;
    bool encoded;
    if (step == 3) {
        out = newBuffer(gpu, 4 * words, &why);
        if (!out || out->contents() == nullptr) {
            std::printf("gpucompute: FAIL no buffer: %s\n", why.c_str());
            return 3;
        }
        canary = canaryFor(words, c);
        uint32_t* p = static_cast<uint32_t*>(out->contents());
        for (uint32_t i = 0; i < words; i++) p[i] = canary;
        uint32_t imm[4] = {c, 0, 0, 0};
        const uint32_t groups[3] = {16, 1, 1};
        encoded = cb->dispatch(kernels::fill_id, groups, nullptr, std::vector<Binding>{Binding{out.get(), 0}}, imm, o);
        std::printf("gpucompute: step 3: fill_id, 16 groups of 64, tail on, c = 0x%x: encoded %s %s\n", c,
                    encoded ? "ok" : "REFUSED", encoded ? "" : cb->error().c_str());
    } else {
        const uint32_t groups[3] = {1, 1, 1};
        encoded = cb->dispatch(kernels::end_only, groups, nullptr, std::vector<Binding>{}, nullptr, o);
        std::printf("gpucompute: step %d: end_only, 1 group, tail %s: encoded %s %s\n", step, step == 1 ? "off" : "on",
                    encoded ? "ok" : "REFUSED", encoded ? "" : cb->error().c_str());
    }
    std::fflush(stdout);
    if (!encoded) return 4;
    signal(SIGALRM, onAlarm);
    alarm(20);
    cb->commit();
    cb->waitUntilCompleted();
    alarm(0);
    const bool done = cb->status() == CommandBuffer::kCompleted;
    std::printf("gpucompute: %s status %d, %lld ns on the GPU, FAULTS %llu -> %llu%s%s\n", done ? "ok  " : "FAIL",
                int(cb->status()), (long long)cb->gpuNanoseconds(), (unsigned long long)faults0,
                (unsigned long long)gpu.faults(), done ? "" : " error: ", done ? "" : cb->error().c_str());
    if (!done) return 5;
    if (step != 3) return 0;

    std::vector<uint32_t> back(words);
    std::memcpy(back.data(), out->contents(), 4 * words);
    uint32_t right = 0, stillCanary = 0, wrong = 0, firstAt = 0, firstValue = 0;
    for (uint32_t i = 0; i < words; i++) {
        if (back[i] == expectedWord(i, c)) right++;
        else if (back[i] == canary) stillCanary++;
        else {
            if (wrong == 0) { firstAt = i; firstValue = back[i]; }
            wrong++;
        }
    }
    std::printf("gpucompute: fill_id: %u of %u words as the kernel computes them, %u still canary, %u wrong", right, words,
                stillCanary, wrong);
    if (wrong) std::printf(" (first at %u: 0x%08x, want 0x%08x)", firstAt, firstValue, expectedWord(firstAt, c));
    std::printf("\n");
    return right == words ? 0 : 6;
}
