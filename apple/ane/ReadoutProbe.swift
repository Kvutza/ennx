import CoreML
import Foundation

enum ANEUnits: String, CaseIterable {
    case cpu
    case gpu
    case ane
    case all

    var coreML: MLComputeUnits {
        switch self {
        case .cpu: .cpuOnly
        case .gpu: .cpuAndGPU
        case .ane: .cpuAndNeuralEngine
        case .all: .all
        }
    }
}

enum ANEWeights: String, CaseIterable {
    case fixed
    case input
}

struct ANEReadoutConfiguration {
    var rows = 128
    var width = 512
    var outputs = 8_192
    var repeats = 7
    var warmups = 3
    var units = ANEUnits.ane
    var weights = ANEWeights.input
}

enum ANEProbeError: Error, CustomStringConvertible {
    case model(String)

    var description: String {
        switch self { case let .model(message): message }
    }
}

private struct Proto {
    var bytes = Data()

    mutating func varint(_ field: UInt64, _ value: UInt64) {
        rawVarint(field << 3)
        rawVarint(value)
    }

    mutating func string(_ field: UInt64, _ value: String) {
        data(field, Data(value.utf8))
    }

    mutating func message(_ field: UInt64, _ body: Proto) {
        data(field, body.bytes)
    }

    mutating func data(_ field: UInt64, _ value: Data) {
        rawVarint((field << 3) | 2)
        rawVarint(UInt64(value.count))
        bytes.append(value)
    }

    private mutating func rawVarint(_ input: UInt64) {
        var value = input
        while value >= 0x80 {
            bytes.append(UInt8(value & 0x7f) | 0x80)
            value >>= 7
        }
        bytes.append(UInt8(value))
    }
}

private func arrayType(shape: [Int]) -> Proto {
    var array = Proto()
    for dimension in shape { array.varint(1, UInt64(dimension)) }
    array.varint(2, 65_552)
    var type = Proto()
    type.message(5, array)
    return type
}

private func feature(_ name: String, shape: [Int]) -> Proto {
    var feature = Proto()
    feature.string(1, name)
    feature.message(3, arrayType(shape: shape))
    return feature
}

private func tensor(_ shape: [Int]) -> Proto {
    var tensor = Proto()
    tensor.varint(1, UInt64(shape.count))
    for dimension in shape { tensor.varint(2, UInt64(dimension)) }
    return tensor
}

private func weights(width: Int, outputs: Int) -> Data {
    var data = Data(capacity: width * outputs * 2)
    for index in 0..<(width * outputs) {
        let signed = Int((UInt64(index) &* 17 &+ 13) % 31) - 15
        var bits = Float16(Float(signed) / 128).bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func model(_ options: ANEReadoutConfiguration) -> Data {
    var description = Proto()
    description.message(1, feature("input", shape: [options.rows, options.width]))
    if options.weights == .input {
        description.message(1, feature("weights", shape: [options.width, options.outputs]))
    }
    description.message(10, feature("output", shape: [options.rows, options.outputs]))

    var layer = Proto()
    layer.string(1, "readout")
    layer.string(2, "input")
    if options.weights == .input { layer.string(2, "weights") }
    layer.string(3, "output")
    layer.message(4, tensor([options.rows, options.width]))
    if options.weights == .input { layer.message(4, tensor([options.width, options.outputs])) }
    layer.message(5, tensor([options.rows, options.outputs]))
    if options.weights == .input {
        layer.message(1_045, Proto())
    } else {
        var weightParams = Proto()
        weightParams.data(2, weights(width: options.width, outputs: options.outputs))
        var innerProduct = Proto()
        innerProduct.varint(1, UInt64(options.width))
        innerProduct.varint(2, UInt64(options.outputs))
        innerProduct.message(20, weightParams)
        layer.message(140, innerProduct)
    }

    var network = Proto()
    network.message(1, layer)
    network.varint(5, 1)

    var model = Proto()
    model.varint(1, 7)
    model.message(2, description)
    model.message(500, network)
    return model.bytes
}

private func device(_ value: MLComputeDevice) -> String {
    switch value {
    case .cpu: "cpu"
    case .gpu: "gpu"
    case .neuralEngine: "ane"
    @unknown default: "unknown"
    }
}

private func milliseconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
}

private func expected(_ channel: Int, _ options: ANEReadoutConfiguration) -> Double {
    var total = 0.0
    for index in 0..<options.width {
        let offset = options.weights == .fixed
            ? channel * options.width + index
            : index * options.outputs + channel
        let signed = Int((UInt64(offset) &* 17 &+ 13) % 31) - 15
        total += Double(signed) / 128
    }
    return total
}

private func check(_ output: MLMultiArray, _ options: ANEReadoutConfiguration) throws -> Double {
    let rows = [0, options.rows - 1]
    let channels = [0, options.outputs / 2, options.outputs - 1]
    var maximum = 0.0
    for row in rows {
        for channel in channels {
            let observed = output[[NSNumber(value: row), NSNumber(value: channel)]].doubleValue
            maximum = max(maximum, abs(observed - expected(channel, options)))
        }
    }
    guard maximum <= 0.01 else {
        throw ANEProbeError.model("readout validation error \(maximum) exceeds 0.01")
    }
    return maximum
}

private func placement(_ plan: MLComputePlan) -> [[String: Any]] {
    guard case let .neuralNetwork(network) = plan.modelStructure else {
        return [["type": "not-neural-network"]]
    }
    return network.layers.map { layer in
        guard let usage = plan.deviceUsage(for: layer) else {
            return ["name": layer.name, "type": layer.type, "preferred": "unknown"]
        }
        return [
            "name": layer.name,
            "type": layer.type,
            "preferred": device(usage.preferred),
            "supported": usage.supported.map(device),
        ]
    }
}

enum ANEReadoutProbe {
    static func run(_ options: ANEReadoutConfiguration, source: String) async throws -> [String: Any] {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ennx-ane-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sourceModel = directory.appendingPathComponent("Readout.mlmodel")
        try model(options)
            .write(to: sourceModel, options: .atomic)
        let compiled = try await MLModel.compileModel(at: sourceModel)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = options.units.coreML
        if #available(macOS 15.0, iOS 18.0, *) {
            configuration.optimizationHints.specializationStrategy = .fastPrediction
        }
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        let loaded = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        let input = try MLMultiArray(
            shape: [NSNumber(value: options.rows), NSNumber(value: options.width)],
            dataType: .float16
        )
        let buffer = input.dataPointer.bindMemory(
            to: Float16.self,
            capacity: options.rows * options.width
        )
        for index in 0..<(options.rows * options.width) { buffer[index] = 1 }
        var features = ["input": MLFeatureValue(multiArray: input)]
        if options.weights == .input {
            let matrix = try MLMultiArray(
                shape: [NSNumber(value: options.width), NSNumber(value: options.outputs)],
                dataType: .float16
            )
            let values = matrix.dataPointer.bindMemory(
                to: Float16.self,
                capacity: options.width * options.outputs
            )
            for index in 0..<(options.width * options.outputs) {
                let signed = Int((UInt64(index) &* 17 &+ 13) % 31) - 15
                values[index] = Float16(Float(signed) / 128)
            }
            features["weights"] = MLFeatureValue(multiArray: matrix)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)

        for _ in 0..<options.warmups { _ = try await loaded.prediction(from: provider) }
        var samples = [Double]()
        var output: MLMultiArray?
        for _ in 0..<options.repeats {
            let start = ContinuousClock.now
            let prediction = try await loaded.prediction(from: provider)
            samples.append(milliseconds(start.duration(to: .now)))
            output = prediction.featureValue(for: "output")?.multiArrayValue
        }
        guard let output else { throw ANEProbeError.model("model produced no output") }
        let first = output[0].doubleValue
        guard first.isFinite else { throw ANEProbeError.model("model output is not finite") }
        let error = try check(output, options)

        let medianMS = samples.sorted()[samples.count / 2]
        let operations = 2.0 * Double(options.rows) * Double(options.width) * Double(options.outputs)
        return [
            "ok": true,
            "backend": "coreml",
            "source": source,
            "units": options.units.rawValue,
            "weights": options.weights.rawValue,
            "rows": options.rows,
            "width": options.width,
            "outputs": options.outputs,
            "repeats": options.repeats,
            "warmups": options.warmups,
            "median-wall-ms": medianMS,
            "tflops": operations / (medianMS / 1_000) / 1e12,
            "first-output": first,
            "maximum-sample-error": error,
            "available-devices": MLComputeDevice.allComputeDevices.map(device),
            "layers": placement(plan),
        ]
    }
}
