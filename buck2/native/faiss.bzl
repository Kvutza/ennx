_CORE = [
    "AutoTune.cpp",
    "Clustering.cpp",
    "SuperKMeans.cpp",
    "IVFlib.cpp",
    "Index.cpp",
    "Index2Layer.cpp",
    "IndexAdditiveQuantizer.cpp",
    "IndexBinary.cpp",
    "IndexBinaryFlat.cpp",
    "IndexBinaryFromFloat.cpp",
    "IndexBinaryHNSW.cpp",
    "IndexBinaryHash.cpp",
    "IndexBinaryIVF.cpp",
    "IndexFlat.cpp",
    "IndexFlatCodes.cpp",
    "IndexHNSW.cpp",
    "IndexIDMap.cpp",
    "IndexIVF.cpp",
    "IndexIVFAdditiveQuantizer.cpp",
    "IndexIVFFlat.cpp",
    "IndexIVFFlatPanorama.cpp",
    "IndexIVFPQ.cpp",
    "IndexIVFFastScan.cpp",
    "IndexIVFAdditiveQuantizerFastScan.cpp",
    "IndexIVFPQFastScan.cpp",
    "IndexIVFPQR.cpp",
    "IndexIVFRaBitQ.cpp",
    "IndexIVFRaBitQFastScan.cpp",
    "IndexIVFSpectralHash.cpp",
    "IndexLSH.cpp",
    "IndexNNDescent.cpp",
    "IndexLattice.cpp",
    "IndexNSG.cpp",
    "IndexPQ.cpp",
    "IndexFastScan.cpp",
    "IndexAdditiveQuantizerFastScan.cpp",
    "IndexIVFIndependentQuantizer.cpp",
    "IndexPQFastScan.cpp",
    "IndexPreTransform.cpp",
    "IndexRaBitQ.cpp",
    "IndexRaBitQFastScan.cpp",
    "IndexRefine.cpp",
    "IndexReplicas.cpp",
    "IndexRowwiseMinMax.cpp",
    "IndexScalarQuantizer.cpp",
    "IndexShards.cpp",
    "IndexShardsIVF.cpp",
    "IndexNeuralNetCodec.cpp",
    "MatrixStats.cpp",
    "MetaIndexes.cpp",
    "VectorTransform.cpp",
    "clone_index.cpp",
    "index_factory.cpp",
    "impl/AdSampling.cpp",
    "impl/AuxIndexStructures.cpp",
    "impl/VisitedTable.cpp",
    "impl/ClusteringInitialization.cpp",
    "impl/ClusteringHelpers.cpp",
    "impl/CodePacker.cpp",
    "impl/CodePackerRaBitQ.cpp",
    "impl/IDSelector.cpp",
    "impl/FaissException.cpp",
    "impl/HNSW.cpp",
    "impl/hnsw/LockVector.cpp",
    "impl/hnsw/MinimaxHeap.cpp",
    "impl/NSG.cpp",
    "impl/PolysemousTraining.cpp",
    "impl/ProductQuantizer.cpp",
    "impl/pq_code_distance/pq_code_distance-generic.cpp",
    "impl/pq_code_distance/IVFPQ_QueryTables.cpp",
    "impl/AdditiveQuantizer.cpp",
    "impl/RaBitQuantizer.cpp",
    "impl/RaBitQuantizerMultiBit.cpp",
    "impl/RaBitQUtils.cpp",
    "impl/ResidualQuantizer.cpp",
    "impl/LocalSearchQuantizer.cpp",
    "impl/ProductAdditiveQuantizer.cpp",
    "impl/ScalarQuantizer.cpp",
    "impl/scalar_quantizer/training.cpp",
    "impl/index_read.cpp",
    "impl/index_write.cpp",
    "impl/io.cpp",
    "impl/kmeans1d.cpp",
    "impl/lattice_Zn.cpp",
    "impl/mapped_io.cpp",
    "impl/fast_scan/fast_scan.cpp",
    "impl/residual_quantizer_encode_steps.cpp",
    "impl/zerocopy_io.cpp",
    "impl/NNDescent.cpp",
    "impl/Panorama.cpp",
    "impl/PanoramaStats.cpp",
    "impl/PdxLayout.cpp",
    "invlists/BlockInvertedLists.cpp",
    "invlists/DirectMap.cpp",
    "invlists/InvertedLists.cpp",
    "invlists/InvertedListsIOHook.cpp",
    "utils/Heap.cpp",
    "utils/NeuralNet.cpp",
    "utils/WorkerThread.cpp",
    "utils/distances.cpp",
    "utils/distances_simd.cpp",
    "utils/extra_distances.cpp",
    "utils/hamming.cpp",
    "utils/partitioning.cpp",
    "utils/quantize_lut.cpp",
    "utils/random.cpp",
    "utils/sorting.cpp",
    "utils/utils.cpp",
    "utils/simd_levels.cpp",
    "utils/distances_fused/distances_fused.cpp",
    "factory_tools.cpp",
]

_NEON = [
    "impl/fast_scan/impl-neon.cpp",
    "impl/scalar_quantizer/sq-neon.cpp",
    "impl/approx_topk/neon.cpp",
    "impl/binary_hamming/neon.cpp",
    "impl/pq_code_distance/neon.cpp",
    "utils/simd_impl/distances_aarch64.cpp",
    "utils/hamming_distance/hamming_neon.cpp",
    "utils/simd_impl/partitioning_neon.cpp",
    "utils/distances_fused/simdlib_based_neon.cpp",
    "utils/simd_impl/rabitq_neon.cpp",
]

_METAL = [
    "MetalResources.mm",
    "MetalIndex.mm",
    "MetalKernels.mm",
    "MetalDistance.mm",
    "MetalFlatKernels.mm",
    "MetalIndexFlat.mm",
    "MetalIndexIVFFlat.mm",
    "impl/MetalIVFFlat.mm",
    "StandardMetalResources.mm",
    "MetalCloner.mm",
    "MetalPythonBridge.mm",
]

def _key(path):
    return path.replace("/", "_").replace(".", "_")

def _unpack(ctx, archive):
    out = ctx.actions.declare_output("openmp", dir = True)
    script = ctx.actions.write(
        "unpack-openmp.sh",
        "#!/bin/sh\nset -eu\nmkdir -p \"$2\"\n/usr/bin/bsdtar -xf \"$1\" -C \"$2\"\n",
        is_executable = True,
    )
    ctx.actions.run(
        [script, archive, out.as_output()],
        category = "faiss_openmp",
    )
    return out

def _compile(ctx, root, omp, path):
    out = ctx.actions.declare_output("obj/{}.o".format(_key(path)))
    cmd = cmd_args(
        "/usr/bin/clang++",
        "-c",
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-mmacosx-version-min=11.0",
        "-Xpreprocessor",
        "-fopenmp",
        "-Wno-deprecated-declarations",
        "-Wno-unknown-pragmas",
        "-DCOMPILE_SIMD_ARM_NEON",
        "-DFINTEGER=int",
    )
    cmd.add(cmd_args(root, format = "-I{}"))
    cmd.add(cmd_args(omp.project("include"), format = "-I{}"))
    cmd.add(root.project("faiss/" + path), "-o", out.as_output())
    ctx.actions.run(cmd, category = "faiss_cxx", identifier = _key(path))
    return out

def _metal(ctx, root, path):
    out = ctx.actions.declare_output("metal_obj/{}.o".format(_key(path)))
    cmd = cmd_args(
        "/usr/bin/clang++",
        "-c",
        "-std=c++20",
        "-O2",
        "-fPIC",
        "-mmacosx-version-min=11.0",
        "-Wno-deprecated-declarations",
        "-DFAISS_METAL_ENABLED=1",
        "-DFAISS_METALLIB_BUILD_PATH=\"\"",
    )
    cmd.add(cmd_args(root, format = "-I{}"))
    cmd.add(root.project("faiss/gpu_metal/" + path), "-o", out.as_output())
    ctx.actions.run(cmd, category = "faiss_objcxx", identifier = _key(path))
    return out

def _archive(ctx, name, objects):
    out = ctx.actions.declare_output(name)
    cmd = cmd_args("/usr/bin/libtool", "-static", "-D", "-o", out.as_output())
    cmd.add(objects)
    ctx.actions.run(cmd, category = "faiss_archive", identifier = name)
    return out

def _shader(ctx, root):
    source = root.project("faiss/gpu_metal/MetalDistance.metal")
    air = ctx.actions.declare_output("MetalDistance.air")
    lib = ctx.actions.declare_output("MetalDistance.metallib")
    ctx.actions.run(
        [
            "/usr/bin/xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            source,
            "-o",
            air.as_output(),
        ],
        category = "faiss_metal_air",
    )
    ctx.actions.run(
        [
            "/usr/bin/xcrun",
            "-sdk",
            "macosx",
            "metallib",
            air,
            "-o",
            lib.as_output(),
        ],
        category = "faiss_metallib",
    )
    return lib

def _impl(ctx):
    root = ctx.attrs.source[DefaultInfo].default_outputs[0]
    package = ctx.attrs.openmp[DefaultInfo].default_outputs[0].project(
        "pkg-llvm-openmp-22.1.8-hc7d1edf_0.tar.zst",
    )
    omp = _unpack(ctx, package)
    core = _archive(
        ctx,
        "libfaiss.a",
        [_compile(ctx, root, omp, path) for path in _CORE + _NEON],
    )
    metal = _archive(
        ctx,
        "libfaiss_metal.a",
        [_metal(ctx, root, path) for path in _METAL],
    )
    metallib = _shader(ctx, root)
    outputs = {
        "faiss": core,
        "metal": metal,
        "metallib": metallib,
        "omp-header": omp.project("include/omp.h"),
        "openmp": omp.project("lib/libomp.dylib"),
    }
    return [
        DefaultInfo(
            default_output = core,
            sub_targets = {
                name: [DefaultInfo(default_output = output)]
                for name, output in outputs.items()
            },
        ),
    ]

faiss_native = rule(
    impl = _impl,
    attrs = {
        "openmp": attrs.dep(),
        "source": attrs.dep(),
    },
)
