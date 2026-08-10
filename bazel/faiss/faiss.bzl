load("@rules_cc//cc:defs.bzl", "objc_library")

_METAL_SRCS = [
    "faiss/gpu_metal/MetalResources.mm",
    "faiss/gpu_metal/MetalIndex.mm",
    "faiss/gpu_metal/MetalKernels.mm",
    "faiss/gpu_metal/MetalDistance.mm",
    "faiss/gpu_metal/MetalFlatKernels.mm",
    "faiss/gpu_metal/MetalIndexFlat.mm",
    "faiss/gpu_metal/MetalIndexIVFFlat.mm",
    "faiss/gpu_metal/impl/MetalIVFFlat.mm",
    "faiss/gpu_metal/StandardMetalResources.mm",
    "faiss/gpu_metal/MetalCloner.mm",
    "faiss/gpu_metal/MetalPythonBridge.mm",
]

def faiss_metal():
    native.genrule(
        name = "metal_shader",
        srcs = ["faiss/gpu_metal/MetalDistance.metal"],
        outs = ["MetalDistance.metallib"],
        cmd = " ".join([
            "/usr/bin/xcrun -sdk macosx metal -c $(location faiss/gpu_metal/MetalDistance.metal)",
            "-o $(@D)/MetalDistance.air &&",
            "/usr/bin/xcrun -sdk macosx metallib $(@D)/MetalDistance.air -o $@",
        ]),
        target_compatible_with = ["@platforms//os:macos"],
    )
    objc_library(
        name = "faiss_metal",
        srcs = _METAL_SRCS,
        hdrs = ["faiss/gpu/GpuIndicesOptions.h"] + native.glob(
            ["faiss/gpu_metal/**/*.h"],
        ),
        copts = [
            "-fno-objc-arc",
            "-std=c++20",
            "-Wno-deprecated-declarations",
            "-DFAISS_METALLIB_BUILD_PATH=\\\"\\\"",
        ],
        defines = [
            "FAISS_METAL_ENABLED=1",
        ],
        includes = ["."],
        sdk_frameworks = [
            "Foundation",
            "Metal",
            "MetalKit",
            "MetalPerformanceShaders",
        ],
        target_compatible_with = ["@platforms//os:macos"],
        deps = [":faiss"],
    )
