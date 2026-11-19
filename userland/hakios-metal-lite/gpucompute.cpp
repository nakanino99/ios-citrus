// gpucompute: the smallest compute dispatches, one step per run. No gpu_recover in this kernel: a hang stays a hang
// until a reboot, so run the steps in order and read each result before the next.
//   1: end_only, 1 group, without the dispatch's own tail
//   2: end_only, 1 group, the whole sequence (tail on)
//   3: fill_id, 16 groups of 64, c = 0x1234: every word checked
//   4: the controls of probe.cpp 5c: another c; 8 groups (half the buffer stays canary); end_only over 16 groups
//   5: fill_id_air (fixture made offline from AIR by ir3_compiler), 16 groups of 64, c from a constant buffer
// exit: 0 all ok, 6 a check failed, 7 hang (alarm), 64 usage, 2 no GPU, 3 no queue/buffer
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "fixtures/end_only.hpp"
#include "fixtures/fill_id.hpp"
#include "fixtures/fill_id_air.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/compute.hpp"

using namespace hakios::metal;

static int failures = 0;

static void onAlarm(int) {
    const char m[] = "gpucompute: no answer after 20 s: the GPU is probably hung (reboot to recover)\n";
    if (write(1, m, sizeof m - 1) < 0) {}
    _exit(7);
}

static void report(bool ok, const std::string& what) {
    std::printf("gpucompute: %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    failures += !ok;
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

struct Readback {
    uint32_t right = 0, canary = 0, wrong = 0, beyondCanary = 0, beyondOther = 0, firstWrongAt = 0, firstWrongValue = 0;
};

// what a buffer holds after a dispatch: the kernel's answer for WRITTEN words, the canary beyond them (probe.cpp)
static Readback readBack(const Buffer& b, uint32_t written, uint32_t c, uint32_t canaryValue) {
    std::vector<uint32_t> back(b.length() / 4);
    std::memcpy(back.data(), b.contents(), b.length());
    Readback r;
    for (uint32_t i = 0; i < back.size(); i++) {
        if (i < written) {
            if (back[i] == expectedWord(i, c)) r.right++;
            else if (back[i] == canaryValue) r.canary++;
            else {
                if (r.wrong == 0) { r.firstWrongAt = i; r.firstWrongValue = back[i]; }
                r.wrong++;
            }
        } else {
            (back[i] == canaryValue ? r.beyondCanary : r.beyondOther)++;
        }
    }
    return r;
}

static std::string describe(const Readback& r, uint32_t written, uint32_t c, uint32_t beyond) {
    char t[300];
    int n = std::snprintf(t, sizeof t, "%u of %u words as the kernel computes them, %u still canary, %u wrong", r.right,
                          written, r.canary, r.wrong);
    if (r.wrong && n > 0 && size_t(n) < sizeof t)
        n += std::snprintf(t + n, sizeof t - n, " (first at %u: 0x%08x, want 0x%08x)", r.firstWrongAt, r.firstWrongValue,
                           expectedWord(r.firstWrongAt, c));
    if (n > 0 && size_t(n) < sizeof t)
        std::snprintf(t + n, sizeof t - n, "; beyond them %u of %u canary", r.beyondCanary, beyond);
    return t;
}

struct Run {
    bool encoded = false;
    std::shared_ptr<CommandBuffer> cb;
};

static Run runKernel(CommandQueue& q, const Kernel& k, uint32_t groups, const uint32_t* local,
                     const std::vector<Binding>& bufs, const uint32_t* imm, const compute::Options& o) {
    const uint32_t g[3] = {groups, 1, 1};
    Run r;
    r.cb = q.commandBuffer();
    r.encoded = r.cb->dispatch(k, g, local, bufs, imm, o);
    if (!r.encoded) return r;
    alarm(20);
    r.cb->commit();
    r.cb->waitUntilCompleted();
    alarm(0);
    return r;
}

static bool completed(const Run& r) { return r.encoded && r.cb->status() == CommandBuffer::kCompleted; }

static std::string statusLine(const Run& r, Gpu& gpu, uint64_t faults0) {
    char t[300];
    if (!r.encoded) {
        std::snprintf(t, sizeof t, "; REFUSED: %s", r.cb->error().c_str());
    } else {
        const bool ok = r.cb->status() == CommandBuffer::kCompleted;
        std::snprintf(t, sizeof t, "; status %d, %lld ns on the GPU, FAULTS %llu -> %llu%s%s", int(r.cb->status()),
                      (long long)r.cb->gpuNanoseconds(), (unsigned long long)faults0, (unsigned long long)gpu.faults(),
                      ok ? "" : " error: ", ok ? "" : r.cb->error().c_str());
    }
    return t;
}

int main(int argc, char** argv) {
    const int step = argc > 1 ? std::atoi(argv[1]) : 0;
    if (step < 1 || step > 5) {
        std::printf("gpucompute: usage: gpucompute 1|2|3|4|5   (1: end_only no tail, 2: end_only whole, 3: fill_id, "
                    "4: controls, 5: fill_id_air)\n");
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
    signal(SIGALRM, onAlarm);
    compute::Options o;   // the whole sequence, as the probe's stage 5
    if (step == 1) o.tail = false;

    if (step <= 2) {
        Run r = runKernel(*queue, kernels::end_only, 1, nullptr, {}, nullptr, o);
        report(completed(r), std::string("step ") + std::to_string(step) + ": end_only, 1 group, tail " +
                                 (step == 1 ? "off" : "on") + statusLine(r, gpu, faults0));
        return failures ? 6 : 0;
    }

    const uint32_t words = 1024, c = 0x00001234;
    auto out = newBuffer(gpu, 4 * words, &why);
    auto cbuf = newBuffer(gpu, 64, &why);
    if (!out || !cbuf || out->contents() == nullptr || cbuf->contents() == nullptr) {
        std::printf("gpucompute: FAIL no buffer: %s\n", why.c_str());
        return 3;
    }
    uint32_t canary = canaryFor(words, c);
    auto prefill = [&](uint32_t value) {
        auto* w = static_cast<uint32_t*>(out->contents());
        for (uint32_t i = 0; i < words; i++) w[i] = value;
    };
    uint32_t imm[4] = {c, 0, 0, 0};

    if (step == 3) {
        prefill(canary);
        Run r = runKernel(*queue, kernels::fill_id, 16, nullptr, {Binding{out.get(), 0}}, imm, o);
        Readback b = readBack(*out, words, c, canary);
        report(completed(r) && b.right == words,
               "step 3: fill_id, 16 groups of 64, c = 0x1234: " + describe(b, words, c, 0) + statusLine(r, gpu, faults0));
    } else if (step == 4) {
        {
            const uint32_t c2 = c + 0x1000, c2canary = canaryFor(words, c2);
            uint32_t imm2[4] = {c2, 0, 0, 0};
            prefill(c2canary);
            Run r = runKernel(*queue, kernels::fill_id, 16, nullptr, {Binding{out.get(), 0}}, imm2, o);
            Readback b = readBack(*out, words, c2, c2canary);
            report(completed(r) && b.right == words,
                   "step 4a, control c = 0x2234: " + describe(b, words, c2, 0) + statusLine(r, gpu, faults0));
        }
        {
            prefill(canary);
            Run r = runKernel(*queue, kernels::fill_id, 8, nullptr, {Binding{out.get(), 0}}, imm, o);
            Readback b = readBack(*out, 512, c, canary);
            report(completed(r) && b.right == 512 && b.beyondCanary == 512,
                   "step 4b, control, 8 groups of the same 1024-word buffer: " + describe(b, 512, c, 512) +
                       statusLine(r, gpu, faults0));
        }
        {
            prefill(canary);
            Run r = runKernel(*queue, kernels::end_only, 16, nullptr, {}, nullptr, o);
            Readback b = readBack(*out, 0, c, canary);
            report(completed(r) && b.beyondCanary == words,
                   "step 4c, control, end_only over 16 groups: " + std::to_string(b.beyondCanary) + " of " +
                       std::to_string(words) + " words canary" + statusLine(r, gpu, faults0));
        }
    } else {
        const uint32_t local[3] = {64, 1, 1};
        static_cast<uint32_t*>(cbuf->contents())[0] = c;
        prefill(canary);
        Run r = runKernel(*queue, kernels::fill_id_air, 16, local, {Binding{out.get(), 0}, Binding{cbuf.get(), 0}},
                          nullptr, o);
        Readback b = readBack(*out, words, c, canary);
        report(completed(r) && b.right == words,
               "step 5: fill_id_air (offline AIR -> NIR -> ir3 fixture), 16 groups of 64, c from a constant buffer: " +
                   describe(b, words, c, 0) + statusLine(r, gpu, faults0));
    }
    return failures ? 6 : 0;
}
