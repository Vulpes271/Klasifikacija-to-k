# Arhiv preteklih poskusov

Za aktivno uporabo glej [README](../README.md). Eksperimentalni izvajalniki niso
privzeta pot; ohranjeni so zaradi reprodukcije objavljenih rezultatov.

| Arhivski izvajalnik | Namen |
| --- | --- |
| `scripts/stage1_florence.sh`, `florence_grounding_llm.sh` | Florence grounding 10/30 slik in primerjava LLM |
| `scripts/experiment200.sh` | Izbor in primerjava 200 slik |
| `scripts/prompt_comparison40.sh`, `prompt_comparison_bf16.sh` | Dva prompta in BF16 razširitev |
| `scripts/lilocbench.sh` | Geometrijsko izbrani pari LILocBench |
| `scripts/thud_*.sh`, `loco_experiment.sh`, `bbox_comparison.sh` | Stari industrijski/slikovni/LLM poskusi |
| `scripts/stage1_describe.sh`, `stage2_reason.sh`, `test_vision*.sh` | Prejšnja generična primerjalna pot |

Pripadajoči moduli `thud_*`, `loco_experiment`, `florence_grounding`,
`florence_grounding_llm`, `prompt_comparison*`, `lilocbench` in prejšnji
`vision_pipeline` ostajajo na istih poteh. Skupne GPU meritve, native validacija,
izris ter Ollama tokenizer so sedaj ponovno uporabljeni iz nevtralnih modulov;
podpisi/arhivski formati rezultatov se ne spreminjajo.

Pred reprodukcijo je treba obnoviti samo potrebne opuščene modele po izvornih
model ID-jih, revizijah in manifestih. Uteži NVILA/Qwen3-VL in manjši Ollama modeli
niso več nameščeni. Obnovitev ni del aktivnega zagona. Delujoča Docker okolja so
ostala. Florence je fizično v `models/florence-2-large`; stara pot je ohranjena
kot simbolna povezava. Neutežni podatki odstranjenih checkpointov so lokalno v
`models/retired-metadata/`, majhne konfiguracije v `maintenance/retired-model-configs/`.

Surovi rezultati, arhivski manifesti, prompte, poročila in objavljene slike so
nedotaknjeni. Lokalnih map z morebitnimi edinstvenimi rezultati nismo odstranjevali.
Izbrisani range bloki in popolni ZIP indeksi LILocBench so ponovljivi prenosi;
izbrane slike, poze, kalibracije, CRC/SHA in izvorni URL-ji so ohranjeni.
