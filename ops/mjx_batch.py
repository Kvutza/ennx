"""Check batched CUDA-Oxide rows through DLPack and MJX on a T4."""

from __future__ import annotations

import cupy as cp
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from ennx.experimental import TurboSearch


def device_batch(search, trials):
    owners = []
    rows = []
    for pointer, size, device in search.device_batch(trials):
        memory = cp.cuda.UnownedMemory(pointer, size, search, device_id=device)
        packed = cp.ndarray(
            (size,), dtype=cp.uint8, memptr=cp.cuda.MemoryPointer(memory, 0)
        )
        owners.append(packed)
        rows.append(jax.dlpack.from_dlpack(packed))
    return jnp.stack(rows), owners


def main() -> None:
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="hinge" type="hinge"/>
          <geom type="capsule" size="0.05 0.5"/>
        </body>
      </worldbody>
      <actuator><motor joint="hinge"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    gpu_model = mjx.put_model(model, impl="jax")
    gpu_data = mjx.make_data(model, impl="jax")
    dimensions = 1_030
    base = np.full((dimensions + 1) // 2, 0x88, dtype=np.uint8)
    leaves = [(0, dimensions, 4, 0.125, 1.0 / dimensions, 0.125)]
    search = TurboSearch(
        base,
        0.0,
        leaves,
        8,
        backend="cuda",
        num_pert=20,
        max_pending=4,
    )
    seeds = np.arange(32, dtype=np.uint64).reshape(4, 8) + 1
    trials = search.ask_batch(seeds, 1, acquisition="thompson")
    packed, owners = device_batch(search, trials)
    codes = jnp.stack((packed & 15, packed >> 4), axis=2).reshape(4, -1)
    controls = jnp.mean(codes.astype(jnp.float32) - 8.0, axis=1, keepdims=True)

    @jax.jit
    def step_batch(ctrl):
        data = jax.tree.map(lambda value: jnp.broadcast_to(value, (4,) + value.shape), gpu_data)
        data = data.replace(ctrl=ctrl)
        return jax.vmap(lambda value: mjx.step(gpu_model, value))(data)

    output = step_batch(controls)
    output.qpos.block_until_ready()
    rewards = np.asarray(-jnp.square(output.qpos[:, 0]), dtype=np.float32)
    accepted = search.tell_batch(trials, rewards)
    assert len(owners) == 4
    assert len(accepted) == 4
    assert np.isfinite(rewards).all()
    print(f"MJX_BATCH ok=true rewards={rewards.tolist()}")


if __name__ == "__main__":
    main()
