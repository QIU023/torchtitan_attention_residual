"""H100 kit, logbook only: the shims of kit_setup/shims plus the kit's flex_no_autotune, for the tower benchmark."""

for _f in ("/workspace/kit_setup/shims/sitecustomize.py", "/workspace/kit_cpmm/local/flex_no_autotune/sitecustomize.py"):
    exec(compile(open(_f).read(), _f, "exec"))
