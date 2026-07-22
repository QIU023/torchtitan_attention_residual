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
