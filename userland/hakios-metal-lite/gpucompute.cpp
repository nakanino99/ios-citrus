// gpucompute: the smallest compute dispatch, one run per variant. end_only is one `end` instruction, one group, no
// buffers: nothing is written, the fence either passes or it does not. Step 1: without the dispatch's own tail (the
// kernel's fence closes the submission, as in gpuping). Step 2: the whole sequence. No gpu_recover in this kernel: a
// hang stays a hang until a reboot, so run step 1, read it, and only then step 2.
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "fixtures/end_only.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/compute.hpp"

using namespace hakios::metal;

static void onAlarm(int) {
    const char m[] = "gpucompute: no answer after 20 s: the GPU is probably hung (reboot to recover)\n";
    if (write(1, m, sizeof m - 1) < 0) {}
    _exit(7);
}

int main(int argc, char** argv) {
    const int step = argc > 1 ? std::atoi(argv[1]) : 0;
    if (step != 1 && step != 2) {
        std::printf("gpucompute: usage: gpucompute 1|2   (1: no dispatch tail, 2: the whole sequence)\n");
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
    const uint32_t groups[3] = {1, 1, 1};
    const bool encoded = cb->dispatch(kernels::end_only, groups, nullptr, std::vector<Binding>{}, nullptr, o);
    std::printf("gpucompute: step %d: end_only, 1 group, tail %s: encoded %s %s\n", step, step == 1 ? "off" : "on",
                encoded ? "ok" : "REFUSED", encoded ? "" : cb->error().c_str());
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
    return done ? 0 : 5;
}
