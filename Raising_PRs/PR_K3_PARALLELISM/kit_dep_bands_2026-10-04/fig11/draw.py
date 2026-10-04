"""Draw a layout (ours_fig_proportions.json) in Figure 11's colours and stack it under the figure's rows."""
import json
from PIL import Image, ImageDraw, ImageFont
COL = {('F', 0): (238, 243, 251), ('F', 1): (175, 210, 242), ('F', 2): (95, 168, 228), ('F', 3): (47, 111, 208),
       ('B', 0): (233, 154, 107), ('B', 1): (225, 126, 82), ('B', 2): (189, 78, 48), ('B', 3): (131, 33, 25),
       ('VF', -1): (213, 232, 212), ('VB', -1): (96, 169, 23)}
ours = {int(k): v for k, v in json.load(open('ours_fig_proportions.json')).items()}
U, H, X0, Y0 = 24, 44, 60, 30
W = X0 + int(80 * U) + 20
img = Image.new('RGB', (W, Y0 + 3 * H + 20), 'white')
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except OSError:
    font = ImageFont.load_default()
d.text((4, 6), 'VisionDepPlan, pp 3 x vp 4, 6 micro-batches, drawn on the figure timeline (ViT fwd 0.5, ViT bwd 1.0)', fill='black', font=font)
for r in range(3):
    y = Y0 + r * H
    d.text((8, y + 12), f'PP{r}', fill='black', font=font)
    for k, st, mb, s, e in ours[r]:
        chunk = -1 if st < 0 else st // 3
        c = COL[(k, chunk)]
        x1, x2 = X0 + s * U, X0 + e * U
        d.rectangle([x1, y, x2, y + H - 2], fill=c, outline='black')
        dark = sum(c) < 400
        tw = d.textlength(str(mb), font=font)
        d.text(((x1 + x2 - tw) / 2, y + 12), str(mb), fill='white' if dark else 'black', font=font)
fig = Image.open('p19-19.png').convert('RGB')
k = 500 / 72
crop = fig.crop((int(85 * k), int(112 * k), int(545 * k), int(165 * k)))
crop = crop.resize((W, int(crop.height * W / crop.width)))
out = Image.new('RGB', (W, crop.height + img.height + 40), 'white')
dd = ImageDraw.Draw(out)
dd.text((4, 4), 'Figure 11 of the K3 report (PDF page 19, rows PP0 to PP2)', fill='black', font=font)
out.paste(crop, (0, 26))
out.paste(img, (0, crop.height + 36))
out.save('fig11_vs_plan.png')
print(out.size)
