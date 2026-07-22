# Overnight PP pressure re-run -- 2026-07-22

torchtitan commit: `f76b3ae9a4ad2d7149c70847710d1af5e767c232`
torch: 2.12.0+cu130

| phase | run | steps | final loss | note |
|---|---|---|---|---|
| A | 48B pp8vp4 naive (seq512) | 300 | 6.52172 | |
| A | 48B pp8vp4 adapter (seq512) | 300 | 6.54421 | |

### L16 sweep SUMMARY (1000 steps)
# PP Pressure Test — 20260722-1018

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | naive | 15.73 | 5.29187 | 175m_attn_res_L16_n8_pp8_vp2_naive |
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | adapter | 11.97 | 5.28376 | 175m_attn_res_L16_n8_pp8_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | naive | 10.37 | 5.44236 | 175m_attn_res_L16_n8_pp4_vp2_naive |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | adapter | 26.15 | 5.43749 | 175m_attn_res_L16_n8_pp4_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | naive | 32.50 | 5.11626 | 175m_attn_res_L16_n8_pp4_vp4_naive |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | adapter | 47.35 | 5.09760 | 175m_attn_res_L16_n8_pp4_vp4_adapter |

### pp4vp2 naive #2 (for naive-vs-naive band)
# PP Pressure Test — 20260722-1422

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | naive | 10.30 | 5.44671 | 175m_attn_res_L16_n8_pp4_vp2_naive |
OVERNIGHT_PP_DONE
BANDS_DONE
