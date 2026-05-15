# PP Pressure Test — 20260511-0906

steps=1000 ngpu=8

| config | PP | VP | LBS | GBS | mode | avg step time (s) | final loss | out dir |
|---|---|---|---|---|---|---|---|---|
| 175m_attn_res_L16_n8 | 8 | 2 | 16 | 16 | adapter | 6.44 | 5.20067 | 175m_attn_res_L16_n8_pp8_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 2 | 8 | 8 | adapter | ? | ? | 175m_attn_res_L16_n8_pp4_vp2_adapter |
| 175m_attn_res_L16_n8 | 4 | 4 | 16 | 16 | adapter | ? | ? | 175m_attn_res_L16_n8_pp4_vp4_adapter |
