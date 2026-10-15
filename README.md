# Klasifikacija fiksnih in premicnih objektov z lokalnim LLM

Za dvostopenjski postopek slika -> opis -> reasoning in nastavljanje promptov,
podrobnosti opisa, obeh modelov ter temperatur glej [README_NASTAVITVE.md](README_NASTAVITVE.md).

## Nov začetek: Docker + majhen resnični nabor + majhen model

Za prvi preizkus uporabi **[README_DOCKER.md](README_DOCKER.md)**.
Vsebuje kopirljive ukaze za Linux: prenos 10 označenih slik NYUv2 (nekaj MB),
Qwen2.5 0.5B (398 MB), CPU ali NVIDIA GPU, zagon in ustavitev.
Ni treba prenašati ali razširjati celotnega THUD arhiva.

**Nov GPU benchmark (Q4/BF16, opis slike → LLM, več parametrov): [README_GPU.md](README_GPU.md).**

Preprost Python 3.11+ projekt za hitro primerjavo lokalnih jezikovnih modelov pri razvrscanju ze oznacenih objektov notranjih prostorov v razrede `fixed`, `movable` in `uncertain`.

Program ne izvaja zaznavanja objektov, segmentacije, SLAM-a ali filtriranja oblaka tock. Vhod je standardiziran JSONL z enim zapisom na sliko.

## Namestitev na Linuxu

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

LLM odjemalec uporablja standardno knjižnico; priprava majhnega NYUv2 vzorca
uporablja Pillow, ki se namesti z `requirements.txt`.

## Lokalni OpenAI-compatible streznik

Program klice en lokalni endpoint oblike `http://127.0.0.1:8080/v1/chat/completions`. Model zazenete zunaj tega programa, na primer z `llama.cpp` ali LM Studio. Primer za `llama.cpp`:

```bash
llama-server -m /pot/do/model.gguf --host 127.0.0.1 --port 8080 --ctx-size 4096 --n-gpu-layers 999
```

`llama.cpp` dokumentacija navaja OpenAI-compatible `/v1/...` endpoint v `llama serve`/server nacinu. Pri LM Studio vklopite lokalni server in v konfiguracijo prepisite njegov URL ter tocno ime modela, kot ga prikaze server.

Konfiguracija:

```bash
cp configs/local-openai-compatible.example.json configs/my-model.json
```

Uredite:

```json
{
  "base_url": "http://127.0.0.1:8080/v1",
  "model": "local-model-name-from-your-server",
  "temperature": 0.0,
  "max_tokens": 512,
  "timeout_seconds": 120,
  "extra_body": {}
}
```

## Vhodni JSONL

Vsaka vrstica predstavlja eno sliko:

```json
{"frame_id":"frame-000001","scene":"canteen","objects":[{"id":1,"label":"chair"},{"id":2,"label":"wall"}]}
```

`id` mora biti enolicen znotraj slike. Razlicne instance istega razreda ostanejo loceni objekti. Dodatna polja, kot so `image_path`, `depth_path` ali `pose_path`, se ohranijo kot metapodatki.

## Prvi preizkus brez LLM

```bash
python -m llm_object_classifier baseline \
  --input examples/demo.jsonl \
  --label-map examples/label_map.csv \
  --output-dir results \
  --run-name demo-baseline
```

Primerjava z rocnimi referencami:

```bash
python -m llm_object_classifier evaluate \
  --results results/demo-baseline/results.jsonl \
  --reference examples/reference.csv \
  --output results/demo-baseline/metrics.json
```

Brez referenc program ne poroca o pravilnosti.

## Test enega lokalnega modela

Najprej zazenite lokalni server, nato:

```bash
python -m llm_object_classifier run \
  --input examples/demo.jsonl \
  --config configs/my-model.json \
  --output-dir results \
  --run-name qwen-demo \
  --limit 10
```

Nadaljevanje prekinjenega testa v istem imeniku:

```bash
python -m llm_object_classifier run \
  --input examples/demo.jsonl \
  --config configs/my-model.json \
  --output-dir results/qwen-demo \
  --resume
```

Pri ponovitvi brez `--resume` se ustvari nov imenik, zato se rezultati istega modela ne prepisujejo.

## Priprava THUD/THUD++ podatkov

Javna dokumentacija THUD/THUD++ potrjuje strukturo `Real_Scenes/.../Label/2D_Object_Detection/frame-*.json`, vendar v dostopnem pregledu ne poda dejanske sheme teh realnih 2D JSON datotek. Dokumentiran je synthetic/Unity format `captures_*.json`, kjer so objekti v `captures[].annotations[].values[]` in imajo `label_name` ter `instance_id`.

Podprt pretvornik za dokumentirani synthetic format:

```bash
python -m llm_object_classifier prepare-thud \
  --input /pot/do/THUD/Synthetic_Scenes/Gym/static/Capture_1/Label \
  --output data/thud_synthetic.jsonl \
  --source-format synthetic-captures \
  --limit 100 \
  --every-n 5
```

Podprto je tudi branje ZIP datoteke brez razsirjanja:

```bash
python -m llm_object_classifier prepare-thud \
  --input /pot/do/RGB-D.zip \
  --output data/thud_synthetic.jsonl \
  --source-format synthetic-captures
```

Za dokoncanje realnega `Label/2D_Object_Detection` pretvornika potrebujem en dejanski `frame-*.json` vzorec iz THUD/THUD++ realne scene. Trenutni program pri `--source-format real-2d` zato jasno javi napako namesto ugibanja strukture oznak.

## Primerjava modelov

Za vsak model zazenite lokalni server posebej in uporabite isti JSONL vhod:

```bash
python -m llm_object_classifier run --input data/thud.jsonl --config configs/model-a.json --output-dir results --run-name model-a
python -m llm_object_classifier run --input data/thud.jsonl --config configs/model-b.json --output-dir results --run-name model-b
python -m llm_object_classifier evaluate --results results/model-a/results.jsonl --reference reference.csv
python -m llm_object_classifier evaluate --results results/model-b/results.jsonl --reference reference.csv
```

Izhodi:

- `results.jsonl`: sprotni zapis, vkljucno s surovim odgovorom in napakami.
- `results.csv`: pregledna tabela po objektih.
- `summary.json`: stevilo uspesnih, neuspesnih in preskocenih slik.
- `metrics.json`: samo, ce podate rocni referencni CSV.

Meritve locijo uspesne in neuspesne zahteve prek polja `status`. Napacnega JSON, manjkajocega ID-ja ali neveljavnega razreda program ne popravlja s tihim ugibanjem.

## Poziv LLM

Poziv je v anglescini in posebej poudari:

> Stationary does not mean permanently fixed.

Prav tako prepove ugibanje, da je omara, polica, pult, miza ali podobno vgrajeno, ce ta podatek ni na voljo.

## Trenutne opombe o modelih in kvantizaciji

Preverjeno 2026-10-13:

- `llama.cpp` server podpira OpenAI-compatible endpoint pod `/v1/...`.
- Za vecje modele na RTX 5090 z 32 GB VRAM zacnite s 4-bit GGUF ali proizvajalcevo QAT 4-bit razlicico, nato po potrebi primerjajte Q5/Q6.
- Qwen3.5-9B in Qwen3.5-27B imata dostopne GGUF kvantizacije pri vec ponudnikih na Hugging Face. Za 27B je smiselno zaceti s Q4.
- Google za Gemma 4 navaja QAT Q4 in GGUF variante, tudi za 26B A4B. Za `google/gemma-4-26B-A4B-it` zacnite s QAT/Q4 ali drugim 4-bit GGUF.
- Za `Ministral-3-14B-Instruct-2512` obstajajo GGUF/Q4 repozitoriji skupnosti; uporabite tocno ime modela iz lokalnega serverja.
- Za `openai/gpt-oss-20b` obstaja `ggml-org/gpt-oss-20b-GGUF`; pri `llama.cpp` uporabite aktualno verzijo in navodila model carda.

Ne izmisljajte imen lokalnih `.gguf` datotek ali imen modela v serverju. Ta projekt v konfiguraciji hrani samo URL, ime, ki ga server sprejme, in parametre klica.

Uporabljeni javni viri za preverjanje:

- THUD overview: https://github.com/jackyzengl/THUD_Dataset_Overview
- THUD++ dataset/paper overview: https://arxiv.org/abs/2412.08096
- llama.cpp server docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Google Gemma docs: https://ai.google.dev/gemma/docs/core.md
- Primeri GGUF repozitorijev: https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF, https://huggingface.co/TheStageAI/Qwen3.5-9B-GGUF, https://huggingface.co/ggml-org/gpt-oss-20b-GGUF, https://huggingface.co/EnlistedGhost/Ministral-3-14B-Instruct-2512-GGUF

## Preverjanje

```bash
python -m unittest discover -s tests
```

Testi pokrijejo:

- veljaven simuliran HTTP odgovor,
- napacen JSON,
- manjkajoc objektni ID,
- baseline in evaluacijo,
- dokumentirani synthetic THUD `captures_*.json` vzorec.

