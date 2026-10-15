"""Local kit only: cc12m-test with every image stretched to 3584 x 3584 (webdataset layout: key.jpg / key.txt / key.json)."""
import io
import sys
import tarfile
from collections import defaultdict

from PIL import Image

src, dst, side = sys.argv[1], sys.argv[2], int(sys.argv[3])
samples = defaultdict(dict)
with tarfile.open(src) as tar:
    for member in tar.getmembers():
        if member.isfile():
            key, ext = member.name.rsplit(".", 1)
            samples[key][ext] = tar.extractfile(member).read()
with tarfile.open(dst, "w") as out:
    for key, files in sorted(samples.items()):
        for ext, data in sorted(files.items()):
            if ext == "jpg":
                image = Image.open(io.BytesIO(data)).convert("RGB").resize((side, side), Image.BICUBIC)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=92)
                data = buffer.getvalue()
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))
print(len(samples), "samples written to", dst)
