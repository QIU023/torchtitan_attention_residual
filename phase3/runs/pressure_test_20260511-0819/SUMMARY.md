# PP Pressure Test — 20260511-0819

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | adapter | 6.43 | 5.19987 | 175m_attn_res_L16_n8_pp8_vp2_adapter |
| 175m_attn_res_L32_n8 | 8 | 2 | 16 | 16 | adapter | 9.98 | 11.76178 | 175m_attn_res_L32_n8_pp8_vp2_adapter |
| 175m_attn_res_L48_n8 | 8 | 2 | 16 | 16 | adapter | 12.74 | 11.76178 | 175m_attn_res_L48_n8_pp8_vp2_adapter |
