"""Check batched CUDA-Oxide rows through DLPack and MJX on a T4."""

from __future__ import annotations

import gc
import math

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from ennx.experimental import ParamBlock, turbo_enn


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
    base = jax.device_put(jnp.zeros(dimensions, dtype=jnp.bfloat16))
    blocks = [ParamBlock(17, 0, dimensions, 0.125, 1.0 / dimensions)]
    search = turbo_enn(
        base,
        0.0,
        blocks,
        8,
        max_pending=4,
    )
    proposals = search.ask(4, 8, 1, 1, acquisition="thompson")
    rows = jax.dlpack.from_dlpack(proposals)
    rows.block_until_ready()
    controls = jnp.mean(rows.astype(jnp.float32), axis=1, keepdims=True)

    @jax.jit
    def step_batch(ctrl):
        data = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (4,) + value.shape), gpu_data
        )
        data = data.replace(ctrl=ctrl)
        return jax.vmap(lambda value: mjx.step(gpu_model, value))(data)

    output = step_batch(controls)
    output.qpos.block_until_ready()
    rewards = -jnp.square(output.qpos[:, 0]).astype(jnp.float32)
    del rows
    gc.collect()
    search.tell(proposals, rewards)
    accepted = search.sync()
    assert len(accepted) == 4
    reward_log = [float(value) for value in jax.device_get(rewards)]
    assert all(math.isfinite(value) for value in reward_log)
    print(f"MJX_BATCH ok=true rewards={reward_log}")


if __name__ == "__main__":
    main()
