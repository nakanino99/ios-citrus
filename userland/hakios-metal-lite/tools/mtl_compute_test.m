// A compute kernel through the real Metal API: library from a .metallib, pipeline, queue, encoder, dispatch, read back.
// Run it under the name metal_test (the plug-in shows the device to a process of that name alone).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    @autoreleasepool {
        const char *path = argc > 1 ? argv[1] : "/var/root/mt/fill3.metallib";
        setvbuf(stdout, NULL, _IONBF, 0);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev ? [[dev name] UTF8String] : "nil");
        if (dev == nil) return 2;

        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&e];
        printf("library: %s%s%s\n", lib ? "made" : "nil", e ? ", error: " : "", e ? [[e description] UTF8String] : "");
        if (lib == nil) return 3;
        printf("functions: %s\n", [[[lib functionNames] description] UTF8String]);

        id<MTLFunction> fn = [lib newFunctionWithName:@"fill3"];
        printf("function fill3: %s\n", fn ? "found" : "nil");
        if (fn == nil) return 4;

        e = nil;
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
        printf("pipeline: %s%s%s\n", ps ? "made" : "nil", e ? ", error: " : "", e ? [[e description] UTF8String] : "");
        if (ps == nil) return 5;
        printf("maxTotalThreadsPerThreadgroup %lu, threadExecutionWidth %lu\n",
               (unsigned long)[ps maxTotalThreadsPerThreadgroup], (unsigned long)[ps threadExecutionWidth]);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        const uint32_t words = 1024;
        id<MTLBuffer> buf = [dev newBufferWithLength:4 * (words + 64) options:MTLResourceStorageModeShared];
        uint32_t *w = (uint32_t *)[buf contents];
        for (uint32_t i = 0; i < words + 64; i++) w[i] = 0xa5a5a5a5u;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        printf("encoder: %s\n", enc ? "made" : "nil");
        if (enc == nil) return 6;
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(16, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("command buffer status %ld%s%s\n", (long)[cb status], [cb error] ? ", error: " : "",
               [cb error] ? [[[cb error] description] UTF8String] : "");

        uint32_t right = 0, tail = 0, firstWrong = ~0u;
        for (uint32_t g = 0; g < words; g++) {
            if (w[g] == g * 3u + 1u) right++;
            else if (firstWrong == ~0u) firstWrong = g;
        }
        for (uint32_t g = words; g < words + 64; g++) tail += w[g] == 0xa5a5a5a5u;
        printf("%s: %u of %u words as computed, %u of 64 canary words untouched", right == words && tail == 64 ? "OK" : "FAIL", right, words, tail);
        if (firstWrong != ~0u) printf(", the first wrong at %u: %08x", firstWrong, w[firstWrong]);
        printf("\n");
        return right == words && tail == 64 ? 0 : 1;
    }
}
