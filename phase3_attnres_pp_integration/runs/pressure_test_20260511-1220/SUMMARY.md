# PP Pressure Test — 20260511-1220

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | naive | 5.98 | 5.19746 | 175m_attn_res_L16_n8_pp8_vp2_naive |
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | adapter | 6.52 | 5.18857 | 175m_attn_res_L16_n8_pp8_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | naive | 3.84 | 5.31943 | 175m_attn_res_L16_n8_pp4_vp2_naive |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | adapter | 4.95 | 5.32476 | 175m_attn_res_L16_n8_pp4_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | naive | 7.76 | 4.97807 | 175m_attn_res_L16_n8_pp4_vp4_naive |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | adapter | ? | ? | 175m_attn_res_L16_n8_pp4_vp4_adapter |
