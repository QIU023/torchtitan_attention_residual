# PP Pressure Test — 20260719-0801

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | naive | 15.98 | 5.27563 | 175m_attn_res_L16_n8_pp8_vp2_naive |
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | adapter | 11.95 | 5.26996 | 175m_attn_res_L16_n8_pp8_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | naive | 10.48 | 5.47948 | 175m_attn_res_L16_n8_pp4_vp2_naive |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 16 | adapter | 26.08 | 5.44676 | 175m_attn_res_L16_n8_pp4_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | naive | 32.65 | 5.11195 | 175m_attn_res_L16_n8_pp4_vp4_naive |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 32 | adapter | 47.80 | 5.10225 | 175m_attn_res_L16_n8_pp4_vp4_adapter |
