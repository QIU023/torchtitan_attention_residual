# Reference files

Downloaded from `https://huggingface.co/moonshotai/Kimi-K3` on 2026-07-27 for
architecture verification:

    configuration_kimi_k3.py  modeling_kimi_linear.py  modeling_kimi_k3.py
    kimi_k3_processor.py      kimi_k3_vision_processing.py
    preprocessor_config.json  model.safetensors.index.json

They are third-party code under the Kimi K3 License and are deliberately NOT
committed. Re-fetch with:

    for f in configuration_kimi_k3.py modeling_kimi_k3.py modeling_kimi_linear.py \
             kimi_k3_vision_processing.py kimi_k3_processor.py \
             preprocessor_config.json model.safetensors.index.json; do
      curl -sfLO "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/$f"
    done

`ref_parity_probe.py` and the key-coverage tests read them from here. The tests
skip cleanly when the files are absent, so CI does not depend on the download.

Loading them under our transformers (5.14.1) needs two stubs -- the reference
targets a release whose `utils.generic` still exported `OutputRecorder` and
`check_model_inputs`. See the loader inside `ref_parity_probe.py`.
