import glob, tqdm
from PIL import Image

files = sorted([
    file for file in glob.glob("/cluster/project/cvg/zuoyue/HoliCity/*/*.jpg")
    if not file.endswith("_imag.jpg")
])

for file in tqdm.tqdm(files):
    img = Image.open(file).resize((4096, 2048), resample=Image.Resampling.LANCZOS)
    img.save("holicity_4096x2048/" + file.split("/")[-1])
