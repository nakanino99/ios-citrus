// gpucompute: the smallest compute dispatches, one step per run. No gpu_recover in this kernel: a hang stays a
// hang until a reboot, so run the steps in order and read each result before the next.
//   1: end_only, 1 group, without the dispatch's own tail
//   2: end_only, 1 group, the whole sequence (tail on)
//   3: fill_id, 16 groups of 64, c = 0x1234: every word checked
//   4: the controls of probe.cpp 5c: another c; 8 groups (half the buffer stays canary); end_only over 16 groups
//   5: fill_id_air (fixture made offline from AIR by ir3_compiler), 16 groups of 64, c from a constant buffer
//   6: chain220_air alone (134 units): reboot first and run this as the boot's first shader (probe.cpp 5a0)
//   7: probe.cpp 5d2: chain160_air, chain220_air, long_chain_air in turn (A610 preloads 64 units, the rest is fetched)
//   8: probe.cpp 5a00: the same kernels with the preload cut to 64 and to 120 units
//   9: probe.cpp 7a: tex_fill_air writes an R32Uint 64x32 image (stib.b.typed.2d at IBO slot 0), then two controls
//  10: probe.cpp 7a: row padding (48x20) and the four-lane RGBA32Uint format
//  11: probe.cpp 7b: tex_copy_air reads a texture through the texture table (isaml t#1) and writes it to an image
// exit: 0 all ok, 6 a check failed, 7 hang (alarm), 64 usage, 2 no GPU, 3 no queue/buffer
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "fixtures/chain160_air.hpp"
#include "fixtures/chain220_air.hpp"
#include "fixtures/end_only.hpp"
#include "fixtures/fill_id.hpp"
#include "fixtures/fill_id_air.hpp"
#include "fixtures/long_chain_air.hpp"
#include "fixtures/tex_copy_air.hpp"
#include "fixtures/tex_fill_air.hpp"
#include "hakios-metal/a6xx.hpp"
#include "hakios-metal/command.hpp"
#include "hakios-metal/compute.hpp"
#include "hakios-metal/texture.hpp"

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

// probe.cpp: STEPS steps of v = rotl((v ^ X) * M, R) from gid ^ c, the processor's answer for the chain kernels
static uint32_t long_chain_reference(uint32_t gid, uint32_t c, unsigned steps) {
    uint32_t v = gid ^ c;
    for (unsigned i = 0; i < steps; i++) {
        const uint32_t x = 0x9E3779B1u * (i + 1);
        const uint32_t m = x | 1, r = 5 + i % 23;
        const uint32_t b = (v ^ (x ^ 0x5BD1E995u)) * m;
        v = (b << r) | (b >> (32 - r));
    }
    return v;
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

// one chain kernel over 16 groups of 64: every word must be the processor's chain of its own gid; with checkTwin the
// chain one step short must disagree almost everywhere (it says the last steps ran), as in probe.cpp 5d2
static void runChain(CommandQueue& queue, Gpu& gpu, uint64_t faults0, Buffer& out, Buffer& cbuf, uint32_t words,
                     uint32_t c, const Kernel& k, unsigned steps, const char* name, uint32_t preloadCap, bool checkTwin) {
    const uint32_t local[3] = {64, 1, 1};
    compute::Options o;
    o.preloadCap = preloadCap;
    static_cast<uint32_t*>(cbuf.contents())[0] = c;
    auto* w = static_cast<uint32_t*>(out.contents());
    const uint32_t canary = canaryFor(words, c);
    for (uint32_t i = 0; i < words; i++) w[i] = canary;
    std::printf("gpucompute: start %s (%u units, %u preloaded by default), preloadCap %u, 16 groups of 64\n", name,
                unsigned(k.instrlen), unsigned(std::min<uint32_t>(k.instrlen, a6xx::kInstrCacheSize)),
                unsigned(preloadCap));
    std::fflush(stdout);
    Run r = runKernel(queue, k, 16, local, {Binding{&out, 0}, Binding{&cbuf, 0}}, nullptr, o);
    const uint32_t* got = static_cast<const uint32_t*>(out.contents());
    uint32_t right = 0, shortTwin = 0, firstWrong = words;
    for (uint32_t g = 0; g < words; g++) {
        if (got[g] == long_chain_reference(g, c, steps)) right++;
        else if (firstWrong == words) firstWrong = g;
        shortTwin += got[g] == long_chain_reference(g, c, steps - 1);
    }
    std::string where;
    if (firstWrong != words) {
        char text[120];
        std::snprintf(text, sizeof text, " (first wrong at %u: got %08x, want %08x)", firstWrong, got[firstWrong],
                      long_chain_reference(firstWrong, c, steps));
        where = text;
    }
    char text[96];
    std::snprintf(text, sizeof text, " (word 0: got %08x, want %08x)", got[0], long_chain_reference(0, c, steps));
    const bool twinOk = !checkTwin || shortTwin < words / 16;
    report(completed(r) && right == words && twinOk,
           std::string("dispatch of ") + name + ": " + std::to_string(right) + " of " + std::to_string(words) +
               " words the " + std::to_string(steps) + "-step chain, " + std::to_string(shortTwin) +
               " the one a step short" + where + text + statusLine(r, gpu, faults0));
}

// --- textures (probe.cpp 7a) -------------------------------------------------------------------------------------------
static std::string hex8(uint32_t v) {
    char t[16];
    std::snprintf(t, sizeof t, "0x%08x", v);
    return t;
}

// a canary word for a texture: one that no pixel can come to hold (probe.cpp)
static uint32_t texCanary(const std::function<bool(uint32_t)>& produced) {
    uint32_t canary = 0xa5a5a5a5u;
    while (produced(canary)) canary += 0x01010101u;
    return canary;
}

// the whole Bo behind a texture, read once in order (write-combined memory)
static std::vector<uint8_t> textureBo(const Texture& t) {
    std::vector<uint8_t> back(t.buffer()->length());
    std::memcpy(back.data(), t.contents(), back.size());
    return back;
}
static void fillBo(const Texture& t, uint32_t word) {
    auto* w = static_cast<uint32_t*>(t.contents());
    for (uint64_t i = 0; i < t.buffer()->length() / 4; i++) w[i] = word;
}

struct PixelRule {
    std::function<bool(uint32_t x, uint32_t y)> written;
    std::function<uint32_t(uint32_t x, uint32_t y, uint32_t lane)> value;
};
struct TexReadback {
    uint32_t covered = 0, right = 0, canary = 0, wrong = 0;
    uint32_t firstWrongX = 0, firstWrongY = 0, firstWrongLane = 0, firstWrongGot = 0, firstWrongWant = 0;
    uint32_t uncovered = 0, uncoveredCanary = 0;
    uint32_t firstTouchedX = 0, firstTouchedY = 0;
    uint64_t padding = 0, paddingUntouched = 0, firstPaddingTouchedAt = 0;
};
static TexReadback compareTexture(const Texture& t, const PixelRule& rule, uint32_t canary) {
    std::vector<uint8_t> bo = textureBo(t);
    const uint32_t cpp = t.info().bytesPerPixel, lanes = cpp / 4, W = t.width(), H = t.height();
    const uint64_t pitch = t.bytesPerRow();
    TexReadback r;
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint32_t px[4] = {};
            std::memcpy(px, bo.data() + y * pitch + uint64_t(x) * cpp, cpp);
            bool isCanary = true;
            for (uint32_t l = 0; l < lanes; l++) isCanary = isCanary && px[l] == canary;
            if (rule.written(x, y)) {
                r.covered++;
                uint32_t badLane = lanes;
                for (uint32_t l = 0; l < lanes && badLane == lanes; l++) {
                    if (px[l] != rule.value(x, y, l)) badLane = l;
                }
                if (badLane == lanes) {
                    r.right++;
                } else if (isCanary) {
                    r.canary++;
                } else {
                    if (r.wrong == 0) {
                        r.firstWrongX = x;
                        r.firstWrongY = y;
                        r.firstWrongLane = badLane;
                        r.firstWrongGot = px[badLane];
                        r.firstWrongWant = rule.value(x, y, badLane);
                    }
                    r.wrong++;
                }
            } else {
                r.uncovered++;
                if (isCanary) {
                    r.uncoveredCanary++;
                } else if (r.uncovered - r.uncoveredCanary == 1) {
                    r.firstTouchedX = x;
                    r.firstTouchedY = y;
                }
            }
        }
    }
    for (uint64_t at = 0; at + 4 <= bo.size(); at += 4) {
        uint64_t row = at / pitch, inRow = at % pitch;
        if (row < H && inRow < uint64_t(W) * cpp) continue;   // inside a pixel
        uint32_t word;
        std::memcpy(&word, bo.data() + at, 4);
        r.padding += 4;
        if (word == canary) r.paddingUntouched += 4;
        else if (r.padding - r.paddingUntouched == 4) r.firstPaddingTouchedAt = at;
    }
    return r;
}
static std::string describeTexture(const TexReadback& r, const char* right) {
    std::string s = std::to_string(r.right) + " of " + std::to_string(r.covered) + " pixels " + right + ", " +
                    std::to_string(r.canary) + " still canary, " + std::to_string(r.wrong) + " wrong";
    if (r.wrong) {
        s += " (first at " + std::to_string(r.firstWrongX) + "," + std::to_string(r.firstWrongY) + " lane " +
             std::to_string(r.firstWrongLane) + ": got " + hex8(r.firstWrongGot) + " expected " + hex8(r.firstWrongWant) + ")";
    }
    if (r.uncovered) {
        s += "; outside the grid " + std::to_string(r.uncoveredCanary) + " of " + std::to_string(r.uncovered) + " pixels canary";
        if (r.uncoveredCanary != r.uncovered)
            s += " (first touched at " + std::to_string(r.firstTouchedX) + "," + std::to_string(r.firstTouchedY) + ")";
    }
    return s;
}
static void reportPadding(const std::string& label, const TexReadback& r) {
    if (r.padding == 0) {
        std::printf("gpucompute: (%s: no padding bytes at this geometry: the rows are the pitch long and fill the Bo)\n",
                    label.c_str());
        std::fflush(stdout);
        return;
    }
    report(r.paddingUntouched == r.padding,
           label + ": row padding: " + std::to_string(r.paddingUntouched) + " of " + std::to_string(r.padding) +
               " bytes untouched" +
               (r.paddingUntouched == r.padding ? "" : " (first touched at byte " + std::to_string(r.firstPaddingTouchedAt) + ")"));
}

// probe.cpp 7a texFill: kernel K over a W x H texture of FORMAT, GROUPS of LOCAL threads. Lane 0 of every covered pixel
// holds (y * grid.x + x) * K + c, lanes 1-3 x, y, 0xdead where the format has them; every uncovered pixel and every byte
// of padding is still the canary. The line "start ..." is printed and flushed first, so a hang names itself.
static void texFill(CommandQueue& queue, Gpu& gpu, uint64_t faults0, Buffer& cbuf, const Kernel& k, PixelFormat format,
                    const char* formatName, uint32_t W, uint32_t H, const uint32_t local[3], uint32_t groupsX,
                    uint32_t groupsY, uint32_t c, const std::string& label) {
    std::printf("gpucompute: start %s\n", label.c_str());
    std::fflush(stdout);
    std::string why;
    TextureDesc d;
    d.format = format;
    d.width = W;
    d.height = H;
    d.usage = kTextureUsageWrite;
    auto t = Texture::make(gpu, d, &why);
    if (t == nullptr) {
        report(false, label + ": a " + std::to_string(W) + "x" + std::to_string(H) + " " + formatName + " texture: " + why);
        return;
    }
    const uint32_t gridW = groupsX * local[0], gridH = groupsY * local[1];
    auto value = [=](uint32_t x, uint32_t y, uint32_t lane) -> uint32_t {
        return lane == 0 ? expectedWord(y * gridW + x, c) : lane == 1 ? x : lane == 2 ? y : 0xdeadu;
    };
    const uint32_t lanes = t->info().bytesPerPixel / 4;
    const uint32_t canary = texCanary([&](uint32_t v) {
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                for (uint32_t l = 0; l < lanes; l++)
                    if (value(x, y, l) == v) return true;
        return false;
    });
    fillBo(*t, canary);
    static_cast<uint32_t*>(cbuf.contents())[0] = c;
    Bindings b;
    b.buffers = {Binding{&cbuf, 0}};   // [[buffer(0)]]: c, a pushed constant
    b.textures = {t.get()};            // [[texture(0)]]: out, the image at IBO slot 0
    const uint32_t groups[3] = {groupsX, groupsY, 1};
    compute::Options o;
    Run r;
    r.cb = queue.commandBuffer();
    r.encoded = r.cb->dispatch(k, groups, local, b, nullptr, o);
    if (r.encoded) {
        alarm(20);
        r.cb->commit();
        r.cb->waitUntilCompleted();
        alarm(0);
    }
    PixelRule rule{[=](uint32_t x, uint32_t y) { return x < gridW && y < gridH; }, value};
    TexReadback rb = compareTexture(*t, rule, canary);
    report(completed(r) && rb.right == rb.covered && rb.covered > 0 && rb.uncoveredCanary == rb.uncovered,
           label + ": " + describeTexture(rb, "as computed") + statusLine(r, gpu, faults0));
    reportPadding(label, rb);
}

// probe.cpp 7b: tex_copy. IN (texture index 1, read through the texture table by isaml t#1) is filled through
// Texture::replace with a pattern injective over the pixels and unlike tex_fill's words; OUT (texture index 0, the
// image at IBO slot 0) is canary; OUT must come out equal to IN, pixel by pixel, and IN untouched.
static void texCopy(CommandQueue& queue, Gpu& gpu, uint64_t faults0, uint32_t W, uint32_t H, const uint32_t local[3],
                    uint32_t groupsX, uint32_t groupsY, const std::string& label) {
    std::printf("gpucompute: start %s\n", label.c_str());
    std::fflush(stdout);
    std::string why;
    TextureDesc din;
    din.format = PixelFormat::kR32Uint;
    din.width = W;
    din.height = H;
    din.usage = kTextureUsageRead;
    TextureDesc dout = din;
    dout.usage = kTextureUsageWrite;
    auto in = Texture::make(gpu, din, &why);
    auto out = in ? Texture::make(gpu, dout, &why) : nullptr;
    if (in == nullptr || out == nullptr) {
        report(false, label + ": two " + std::to_string(W) + "x" + std::to_string(H) + " R32Uint textures: " + why);
        return;
    }
    // 0x01000193 is odd (a bijection on 32-bit words), the xor keeps the words apart from tex_fill's
    auto inWord = [=](uint32_t x, uint32_t y) -> uint32_t { return ((y * W + x) * 0x01000193u) ^ 0x6a09e667u; };
    std::vector<uint32_t> host(size_t(W) * H);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) host[size_t(y) * W + x] = inWord(x, y);
    const uint32_t canary = texCanary([&](uint32_t v) {
        for (uint32_t w : host)
            if (w == v) return true;
        return false;
    });
    fillBo(*in, canary ^ 0xffffffffu);   // in's own padding: something else, so a read of it would show
    fillBo(*out, canary);
    std::string whyReplace;
    if (!in->replace(0, 0, W, H, host.data(), uint64_t(W) * 4, &whyReplace)) {
        report(false, label + ": Texture::replace of the input: " + whyReplace);
        return;
    }
    Bindings b;
    b.textures = {out.get(), in.get()};   // [[texture(0)]] = out (the image), [[texture(1)]] = in (sampled)
    const uint32_t groups[3] = {groupsX, groupsY, 1};
    const uint32_t gridW = groupsX * local[0], gridH = groupsY * local[1];
    compute::Options o;
    Run r;
    r.cb = queue.commandBuffer();
    r.encoded = r.cb->dispatch(kernels::tex_copy_air, groups, local, b, nullptr, o);
    if (r.encoded) {
        alarm(20);
        r.cb->commit();
        r.cb->waitUntilCompleted();
        alarm(0);
    }
    PixelRule rule{[=](uint32_t x, uint32_t y) { return x < gridW && y < gridH; },
                   [=](uint32_t x, uint32_t y, uint32_t) { return inWord(x, y); }};
    TexReadback rb = compareTexture(*out, rule, canary);
    report(completed(r) && rb.right == rb.covered && rb.covered > 0 && rb.uncoveredCanary == rb.uncovered,
           label + ": " + describeTexture(rb, "equal to the input") + statusLine(r, gpu, faults0));
    reportPadding(label, rb);
    // the input is read only: every pixel still the pattern (its padding is not checked: nothing wrote it but us)
    PixelRule all{[](uint32_t, uint32_t) { return true; }, [=](uint32_t x, uint32_t y, uint32_t) { return inWord(x, y); }};
    TexReadback rin = compareTexture(*in, all, canary);
    report(rin.right == rin.covered, label + ": the input afterwards: " + std::to_string(rin.right) + " of " +
                                         std::to_string(rin.covered) + " pixels still the pattern");
}

int main(int argc, char** argv) {
    const int step = argc > 1 ? std::atoi(argv[1]) : 0;
    if (step < 1 || step > 11) {
        std::printf("gpucompute: usage: gpucompute 1..11   (1: end_only no tail, 2: end_only whole, 3: fill_id, "
                    "4: controls, 5: fill_id_air, 6: chain220 alone, 7: chains, 8: preload cuts, 9-10: tex_fill, 11: tex_copy)\n");
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
    } else if (step == 5) {
        const uint32_t local[3] = {64, 1, 1};
        static_cast<uint32_t*>(cbuf->contents())[0] = c;
        prefill(canary);
        Run r = runKernel(*queue, kernels::fill_id_air, 16, local, {Binding{out.get(), 0}, Binding{cbuf.get(), 0}},
                          nullptr, o);
        Readback b = readBack(*out, words, c, canary);
        report(completed(r) && b.right == words,
               "step 5: fill_id_air (offline AIR -> NIR -> ir3 fixture), 16 groups of 64, c from a constant buffer: " +
                   describe(b, words, c, 0) + statusLine(r, gpu, faults0));
    } else if (step == 6) {
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain220_air, 220, "chain220_air", 0, false);
    } else if (step == 7) {
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain160_air, 160, "chain160_air", 0, true);
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain220_air, 220, "chain220_air", 0, true);
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::long_chain_air, 700, "long_chain_air", 0, true);
    } else if (step == 8) {
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain160_air, 160, "chain160_air", 64, false);
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain220_air, 220, "chain220_air", 64, false);
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::chain220_air, 220, "chain220_air", 120, false);
        runChain(*queue, gpu, faults0, *out, *cbuf, words, c, kernels::long_chain_air, 700, "long_chain_air", 64, false);
    } else if (step <= 10) {
        const uint32_t local88[3] = {8, 8, 1}, local84[3] = {8, 4, 1};
        const uint32_t c1 = 0x5678, c2 = 0x6789;
        const Kernel& k = kernels::tex_fill_air;
        if (step == 9) {
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kR32Uint, "R32Uint", 64, 32, local88, 8, 4, c1,
                    "tex_fill R32Uint 64x32, (8,4,1) groups of (8,8,1), c = 0x5678");
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kR32Uint, "R32Uint", 64, 32, local88, 4, 4, c1,
                    "control, tex_fill R32Uint 64x32 over the left half, (4,4,1) groups");
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kR32Uint, "R32Uint", 64, 32, local88, 8, 4, c2,
                    "control, tex_fill R32Uint 64x32, c = 0x6789");
        } else {
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kR32Uint, "R32Uint", 48, 20, local84, 6, 5, c1,
                    "tex_fill R32Uint 48x20, (6,5,1) groups of (8,4,1), 192-byte rows at a pitch of 256");
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kRGBA32Uint, "RGBA32Uint", 64, 32, local88, 8, 4, c1,
                    "tex_fill RGBA32Uint 64x32, (8,4,1) groups of (8,8,1), all four lanes");
            texFill(*queue, gpu, faults0, *cbuf, k, PixelFormat::kRGBA32Uint, "RGBA32Uint", 48, 20, local84, 6, 5, c1,
                    "tex_fill RGBA32Uint 48x20, (6,5,1) groups of (8,4,1), 768-byte rows at a pitch of 1024");
        }
    } else {
        const uint32_t local88[3] = {8, 8, 1}, local84[3] = {8, 4, 1};
        texCopy(*queue, gpu, faults0, 64, 32, local88, 8, 4,
                "tex_copy R32Uint 64x32, in at texture 1 (isaml), out at texture 0 (IBO slot 0), (8,4,1) groups of (8,8,1)");
        texCopy(*queue, gpu, faults0, 64, 32, local88, 4, 4, "control, tex_copy R32Uint 64x32 over the left half, (4,4,1) groups");
        texCopy(*queue, gpu, faults0, 48, 20, local84, 6, 5,
                "tex_copy R32Uint 48x20, (6,5,1) groups of (8,4,1), 192-byte rows at a pitch of 256");
        {
            // and Texture::replace refuses a region past the edge, by name
            TextureDesc d;
            d.format = PixelFormat::kR32Uint;
            d.width = 8;
            d.height = 8;
            auto tx = Texture::make(gpu, d, &why);
            uint32_t eight[8] = {};
            std::string whyReplace;
            bool accepted = tx && tx->replace(7, 0, 2, 1, eight, 32, &whyReplace);
            report(tx && !accepted && !whyReplace.empty(),
                   std::string("control, Texture::replace of 2 pixels starting at the last column of 8x8: ") +
                       (accepted ? "ACCEPTED a region past the width" : "refused: " + whyReplace));
        }
    }
    return failures ? 6 : 0;
}
