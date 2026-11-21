// One triangle through the real Metal API into a 64x64 RGBA8 texture, read back. Run it under the name metal_test.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    @autoreleasepool {
        const char *path = argc > 1 ? argv[1] : "/var/root/mr/tri.metallib";
        setvbuf(stdout, NULL, _IONBF, 0);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev ? [[dev name] UTF8String] : "nil");
        if (dev == nil) return 2;

        NSError *e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&e];
        printf("library: %s%s%s\n", lib ? "made" : "nil", e ? ", error: " : "", e ? [[e description] UTF8String] : "");
        if (lib == nil) return 3;
        printf("functions: %s\n", [[[lib functionNames] description] UTF8String]);
        id<MTLFunction> vs = [lib newFunctionWithName:@"tri_vs"];
        id<MTLFunction> fs = [lib newFunctionWithName:@"tri_fs"];
        printf("functions found: vertex %s, fragment %s\n", vs ? "yes" : "no", fs ? "yes" : "no");
        if (vs == nil || fs == nil) return 4;

        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = vs;
        pd.fragmentFunction = fs;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        e = nil;
        id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
        printf("pipeline: %s%s%s\n", ps ? "made" : "nil", e ? ", error: " : "", e ? [[e description] UTF8String] : "");
        if (ps == nil) return 5;

        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:64 height:64 mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
        printf("texture: %s\n", tex ? "made" : "nil");
        if (tex == nil) return 6;

        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 1.0, 1.0);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        printf("encoder: %s\n", enc ? "made" : "nil");
        if (enc == nil) return 7;
        [enc setRenderPipelineState:ps];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("command buffer status %ld%s%s\n", (long)[cb status], [cb error] ? ", error: " : "",
               [cb error] ? [[[cb error] description] UTF8String] : "");

        uint8_t px[64 * 64 * 4];
        memset(px, 0x77, sizeof px);
        [tex getBytes:px bytesPerRow:64 * 4 fromRegion:MTLRegionMake2D(0, 0, 64, 64) mipmapLevel:0];
        unsigned inside = 0, outside = 0, insideTotal = 0, outsideTotal = 0, bad = 0, firstX = 0, firstY = 0;
        for (unsigned y = 0; y < 64; y++) {
            for (unsigned x = 0; x < 64; x++) {
                const uint8_t *p = px + 4 * (y * 64 + x);
                if (x == y) continue;   // on the diagonal: the top-left rule, not counted
                bool want_in = x < y;
                bool is_in = abs((int)p[0] - 0xff) <= 1 && abs((int)p[1] - 0x80) <= 1 && abs((int)p[2] - 0x40) <= 1 && p[3] == 0xff;
                bool is_out = p[0] == 0 && p[1] == 0 && p[2] == 0xff && p[3] == 0xff;
                if (want_in) { insideTotal++; inside += is_in; } else { outsideTotal++; outside += is_out; }
                if ((want_in && !is_in) || (!want_in && !is_out)) { if (bad++ == 0) { firstX = x; firstY = y; } }
            }
        }
        printf("%s: %u of %u interior pixels the colour, %u of %u exterior pixels the clear colour", (bad == 0 && insideTotal == 2016 && outsideTotal == 2016) ? "OK" : "FAIL", inside, insideTotal, outside, outsideTotal);
        if (bad) {
            const uint8_t *p = px + 4 * (firstY * 64 + firstX);
            printf(", the first wrong at (%u, %u): %02x %02x %02x %02x", firstX, firstY, p[0], p[1], p[2], p[3]);
        }
        printf("\n");
        return bad == 0 ? 0 : 1;
    }
}
