def _suite_impl(ctx):
    outputs = []
    for dep in ctx.attrs.test_deps:
        info = dep[DefaultInfo]
        outputs.extend(info.default_outputs)
        outputs.extend(info.other_outputs)
        test = dep.get(ExternalRunnerTestInfo)
        if test != None:
            # Shell tests have no default output. Their executable and resources
            # live in the test provider, which must also be prepared by build.
            outputs.append(cmd_args(test.command, hidden = test.env.values()))
    return [DefaultInfo(other_outputs = outputs)]

_suite = rule(
    impl = _suite_impl,
    attrs = {"test_deps": attrs.list(attrs.dep())},
)

def test_suite(name, tests, **kwargs):
    _suite(name = name, tests = tests, test_deps = tests, **kwargs)
