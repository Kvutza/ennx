# iPhone compute source

The iOS worker runs bounded ENNX kernels on the phone GPU and returns measurements over the local network. Its execution boundary is an independent candidate evaluator: resident state stays on the phone and only job identity and results cross the network. It does not split a latency-sensitive forward pass with the Mac.

```sh
./ennx iphone devices
./ennx iphone build
./ennx iphone deploy door --team "$ENNX_APPLE_TEAM"
./ennx iphone probe door.coredevice.local
./ennx iphone ane door.coredevice.local --rows 4096
./ennx iphone web
./ennx iphone web --workload readout
./ennx iphone web --run .cache/ennx/runs/context-loop/RUN
```

The worker listens on TCP port `47123` and advertises `_ennx._tcp`. Keep the app in the foreground during experiments; iOS suspends ordinary GPU work after the app enters the background.

`iphone probe` is the first implemented job. It runs the resident Rademacher proposal kernel, checks that its sampled output changed, and reports device and wall time, effective memory traffic, power state, and thermal state. This measurement determines whether the phone can contribute candidate throughput before full model scoring is added behind the same protocol.

`iphone web` is the no-install path. It publishes a one-shot worker page over Tailscale HTTPS. Open the printed URL in iPhone Safari and keep it in the foreground. Without a run it measures the resident proposal kernel. With `--run`, it sends that run's generated candidate and reference continuation to the phone, evaluates the registered token-accuracy objective on WebGPU, and requires exact agreement with the Rust objective before accepting the result. This path uses WGSL and browser-visible GPU limits; the native path uses the Metal kernel directly.

`iphone ane` runs the same FP16 readout probe as `./ennx ane probe`. Runtime-input weights model an ENNX candidate without recompiling Core ML. The result includes Core ML's preferred device and numerical validation. This command requires the native worker because Safari does not expose ANE.
