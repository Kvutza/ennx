import Foundation

private enum ArgumentError: Error, CustomStringConvertible {
    case invalid(String)

    var description: String {
        switch self { case let .invalid(message): message }
    }
}

private struct Arguments {
    var configuration = ANEReadoutConfiguration()

    static func parse() throws -> Arguments {
        var result = Arguments()
        var values = Array(CommandLine.arguments.dropFirst())
        while !values.isEmpty {
            let flag = values.removeFirst()
            guard !values.isEmpty else { throw ArgumentError.invalid("missing value for \(flag)") }
            let value = values.removeFirst()
            switch flag {
            case "--rows": result.configuration.rows = try positive(value, flag)
            case "--width": result.configuration.width = try positive(value, flag)
            case "--outputs": result.configuration.outputs = try positive(value, flag)
            case "--repeats": result.configuration.repeats = try positive(value, flag)
            case "--warmups": result.configuration.warmups = try nonnegative(value, flag)
            case "--units":
                guard let units = ANEUnits(rawValue: value) else {
                    throw ArgumentError.invalid("--units must be cpu, gpu, ane, or all")
                }
                result.configuration.units = units
            case "--weights":
                guard let weights = ANEWeights(rawValue: value) else {
                    throw ArgumentError.invalid("--weights must be fixed or input")
                }
                result.configuration.weights = weights
            default: throw ArgumentError.invalid("unknown argument \(flag)")
            }
        }
        return result
    }

    private static func positive(_ value: String, _ flag: String) throws -> Int {
        guard let number = Int(value), number > 0 else {
            throw ArgumentError.invalid("\(flag) must be positive")
        }
        return number
    }

    private static func nonnegative(_ value: String, _ flag: String) throws -> Int {
        guard let number = Int(value), number >= 0 else {
            throw ArgumentError.invalid("\(flag) must be nonnegative")
        }
        return number
    }
}

@main
private struct Main {
    static func main() async {
        do {
            let report = try await ANEReadoutProbe.run(Arguments.parse().configuration, source: "mac")
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } catch {
            fputs("ennx ane: \(error)\n", stderr)
            exit(1)
        }
    }
}
