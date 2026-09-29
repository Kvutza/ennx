import Combine
import Foundation
import Metal
import Network
import SwiftUI
import UIKit

private let workerPort = NWEndpoint.Port(rawValue: 47_123)!

@main
struct EnnxComputeApp: App {
    @StateObject private var worker = Worker()

    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 14) {
                Text("ENNX Compute")
                    .font(.largeTitle.bold())
                Text(worker.state)
                    .font(.headline)
                Text("TCP 47123 · _ennx._tcp")
                    .font(.system(.body, design: .monospaced))
                Text(worker.result)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                Spacer()
            }
            .padding(24)
            .task { worker.start() }
        }
    }
}

final class Worker: ObservableObject {
    @Published var state = "Starting Metal worker"
    @Published var result = "Waiting for ./ennx iphone probe"

    private let queue = DispatchQueue(label: "dev.ennx.compute")
    private var listener: NWListener?
    private var probe: ProposalProbe?

    func start() {
        guard listener == nil else { return }
        do {
            probe = try ProposalProbe()
            let listener = try NWListener(using: .tcp, on: workerPort)
            listener.service = NWListener.Service(name: UIDevice.current.name, type: "_ennx._tcp")
            listener.newConnectionHandler = { [weak self] connection in
                self?.accept(connection)
            }
            listener.stateUpdateHandler = { [weak self] state in
                DispatchQueue.main.async {
                    switch state {
                    case .ready:
                        self?.state = "Ready on port 47123"
                    case let .failed(error):
                        self?.state = "Listener failed: \(error)"
                    case .cancelled:
                        self?.state = "Stopped"
                    default:
                        break
                    }
                }
            }
            listener.start(queue: queue)
            self.listener = listener
        } catch {
            state = "Startup failed: \(error)"
        }
    }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, accumulated: Data())
    }

    private func receive(_ connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self] chunk, _, complete, error in
            var data = accumulated
            if let chunk { data.append(chunk) }
            if let newline = data.firstIndex(of: 10) {
                self?.handle(Data(data[..<newline]), connection: connection)
            } else if complete || error != nil {
                self?.send(["ok": false, "error": "request ended before newline"], on: connection)
            } else {
                self?.receive(connection, accumulated: data)
            }
        }
    }

    private func handle(_ data: Data, connection: NWConnection) {
        do {
            guard
                let request = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                request["protocol"] as? String == "ennx.iphone.v1",
                let command = request["command"] as? String
            else {
                throw WorkerError.request
            }
            if command == "ane-readout" {
                try ane(request, connection: connection)
                return
            }
            guard
                command == "proposal-probe",
                let coordinates = number(request["coordinates"]),
                let repeats = number(request["repeats"])
            else { throw WorkerError.request }
            guard coordinates > 0, repeats > 0, repeats <= 100 else {
                throw WorkerError.range
            }
            guard let probe else { throw WorkerError.metal }
            let report = try probe.run(coordinates: coordinates, repeats: UInt32(repeats))
            send(report, on: connection)
            let encoded = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted])
            DispatchQueue.main.async { [weak self] in
                self?.result = String(decoding: encoded, as: UTF8.self)
            }
        } catch {
            send(["ok": false, "error": String(describing: error)], on: connection)
        }
    }

    private func ane(_ request: [String: Any], connection: NWConnection) throws {
        guard
            let rows = number(request["rows"]).flatMap(Int.init(exactly:)),
            let width = number(request["width"]).flatMap(Int.init(exactly:)),
            let outputs = number(request["outputs"]).flatMap(Int.init(exactly:)),
            let repeats = number(request["repeats"]).flatMap(Int.init(exactly:)),
            let warmups = number(request["warmups"]).flatMap(Int.init(exactly:)),
            let unitsName = request["units"] as? String,
            let units = ANEUnits(rawValue: unitsName),
            let weightsName = request["weights"] as? String,
            let weights = ANEWeights(rawValue: weightsName),
            rows > 0, width > 0, outputs > 0, repeats > 0, warmups >= 0
        else { throw WorkerError.request }
        let options = ANEReadoutConfiguration(
            rows: rows,
            width: width,
            outputs: outputs,
            repeats: repeats,
            warmups: warmups,
            units: units,
            weights: weights
        )
        Task {
            do {
                let report = try await ANEReadoutProbe.run(options, source: "iphone")
                self.send(report, on: connection)
                let encoded = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted])
                await MainActor.run { self.result = String(decoding: encoded, as: UTF8.self) }
            } catch {
                self.send(["ok": false, "error": String(describing: error)], on: connection)
            }
        }
    }

    private func send(_ value: [String: Any], on connection: NWConnection) {
        do {
            let data = try JSONSerialization.data(withJSONObject: value)
            connection.send(content: data, completion: .contentProcessed { _ in connection.cancel() })
        } catch {
            connection.cancel()
        }
    }

    private func number(_ value: Any?) -> UInt64? {
        (value as? NSNumber)?.uint64Value
    }
}

private enum WorkerError: Error, CustomStringConvertible {
    case request
    case range
    case metal
    case allocation
    case execution(String)

    var description: String {
        switch self {
        case .request: "unsupported ENNX request"
        case .range: "coordinates or repeats are outside the worker limits"
        case .metal: "Metal is unavailable"
        case .allocation: "the requested resident buffers exceed this device"
        case let .execution(message): message
        }
    }
}

private struct ProposalWire {
    var seed: UInt64
    var round: UInt32
    var radius: Float
    var coordinates: UInt64
}

private final class ProposalProbe {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    init() throws {
        guard
            let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue(),
            let library = device.makeDefaultLibrary(),
            let function = library.makeFunction(name: "ennx_proposal")
        else { throw WorkerError.metal }
        self.device = device
        self.queue = queue
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func run(coordinates: UInt64, repeats: UInt32) throws -> [String: Any] {
        let bytes = coordinates.multipliedReportingOverflow(by: 2)
        guard
            !bytes.overflow,
            bytes.partialValue <= UInt64(Int.max),
            bytes.partialValue <= UInt64(device.maxBufferLength),
            bytes.partialValue * 2 <= ProcessInfo.processInfo.physicalMemory
        else { throw WorkerError.allocation }
        let length = Int(bytes.partialValue)
        guard
            let base = device.makeBuffer(length: length, options: .storageModeShared),
            let candidate = device.makeBuffer(length: length, options: .storageModeShared)
        else { throw WorkerError.allocation }
        memset(base.contents(), 0, length)

        var samples = [Double]()
        var wallSamples = [Double]()
        for round in 0..<repeats {
            var wire = ProposalWire(
                seed: 0x243f_6a88_85a3_08d3,
                round: round,
                radius: 0.0078125,
                coordinates: coordinates
            )
            guard
                let command = queue.makeCommandBuffer(),
                let encoder = command.makeComputeCommandEncoder()
            else { throw WorkerError.metal }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(base, offset: 0, index: 0)
            encoder.setBuffer(candidate, offset: 0, index: 1)
            encoder.setBytes(&wire, length: MemoryLayout<ProposalWire>.stride, index: 2)
            let vectors = Int((coordinates + 3) / 4)
            let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreads(
                MTLSize(width: vectors, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
            encoder.endEncoding()
            let start = ContinuousClock.now
            command.commit()
            command.waitUntilCompleted()
            let wall = start.duration(to: .now)
            guard command.status == .completed else {
                throw WorkerError.execution(command.error?.localizedDescription ?? "Metal command failed")
            }
            samples.append((command.gpuEndTime - command.gpuStartTime) * 1_000)
            wallSamples.append(milliseconds(wall))
        }

        let count = min(Int(coordinates), 4096)
        let values = candidate.contents().assumingMemoryBound(to: UInt16.self)
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        var changed = 0
        for index in 0..<count {
            let value = values[index]
            changed += value == 0 ? 0 : 1
            hash = (hash ^ UInt64(value)) &* 0x1000_0000_01b3
        }
        guard changed == count else {
            throw WorkerError.execution("proposal validation found unchanged coordinates")
        }

        samples.sort()
        wallSamples.sort()
        let deviceMS = samples[samples.count / 2]
        let wallMS = wallSamples[wallSamples.count / 2]
        let trafficGB = Double(bytes.partialValue) * 2 / 1_000_000_000
        return [
            "ok": true,
            "protocol": "ennx.iphone.v1",
            "source": "iphone",
            "backend": "metal",
            "kernel": "resident-rademacher-proposal",
            "device": device.name,
            "system": UIDevice.current.systemName + " " + UIDevice.current.systemVersion,
            "physical-memory": ProcessInfo.processInfo.physicalMemory,
            "max-buffer-length": device.maxBufferLength,
            "low-power": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "thermal": thermal(ProcessInfo.processInfo.thermalState),
            "coordinates": coordinates,
            "repeats": repeats,
            "median-device-ms": deviceMS,
            "median-wall-ms": wallMS,
            "effective-gb-s": trafficGB / (deviceMS / 1_000),
            "sampled-coordinates": count,
            "sample-hash": String(format: "%016llx", hash),
        ]
    }

    private func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }

    private func thermal(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}
