# K2.5 对照（10-18）

- `compare_k25.py`：titan 的视频分段对照发布的 Kimi K2.5 processor 代码（用 K3 的 `preprocessor_config.json`）。
  - Part A：在 2888 组（帧数、fps）上比采样帧号和整段 prompt 文本，元数据是假的，只跑算术。
  - Part B：PyAV 解码真实视频一次，同一批帧分别喂给 titan 的分段函数和 K2.5 的 `split_video_chunks` + `preprocess`，比帧号、prompt、grid、像素。
- `compare_k3_image.py`：titan 的图片路径对照发布的 K3 图片 processor，用来确认缩放差异在 main 的图片路径上也有。
- `make_videos.sh`：生成 Part B 的五段视频（cc12m-test 的两张照片在 `tests/assets/cc12m_test/cc12m-train-0000.tar` 里）。
- 运行需要：K2.5 的远程代码 `kimi_k25_vision_processing.py`、`media_utils.py`（放进名为 `k25ref` 的包），K3 的 `kimi_k3_vision_processing.py`、`media_utils.py`（包名 `k3ref`），`transformers`（`scratchpad/parity_pins`），`pydantic`，PyAV。这些文件不进 logbook。
