# Colab CUDA

Start a Colab T4 session:

```sh
uv tool install google-colab-cli --with jupyter-kernel-client==0.14.0 --force
colab new -s ennx-cuda --gpu T4
```

Open its shell. With `jj` installed on the VM, clone the project:

```sh
colab console -s ennx-cuda
cd /content
jj git clone --depth 1 --branch main https://github.com/Kvutza/ennx.git ennx
cd ennx
```

Run the [CUDA setup and checks](../cuda/README.md) from `/content/ennx`.

From your local terminal, stop the session when finished:

```sh
colab stop -s ennx-cuda
```
