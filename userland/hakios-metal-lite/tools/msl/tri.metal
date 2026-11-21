#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
};

// vertices 0, 1, 2: (-1, -1), (1, -1), (-1, 1): the lower-left half of the target
vertex VOut tri_vs(uint vid [[vertex_id]])
{
    VOut o;
    o.pos = float4(vid == 1 ? 1.0 : -1.0, vid == 2 ? 1.0 : -1.0, 0.0, 1.0);
    return o;
}

fragment float4 tri_fs()
{
    return float4(1.0, 0.5, 0.25, 1.0);
}
