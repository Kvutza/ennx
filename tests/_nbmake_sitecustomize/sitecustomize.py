try:
    import jupyter_client.localinterfaces as _li

    def _disableaddrs() -> None:
        raise ImportError(
            "disabled psutil-based IP discovery for nbmake in this environment"
        )

    _li._load_ips_psutil = _disableaddrs
except Exception:
    pass
