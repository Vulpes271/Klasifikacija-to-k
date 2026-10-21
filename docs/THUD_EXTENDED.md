# NVILA-8B in Florence-2-large: ločena GPU izvajalnika

Nadaljevanje uporablja iste RGB bajte kot 2. del: 50 slik trgovine in 50 agtea.
Lokalni `results/thud-part2/inputs/` in stari rezultati so predpogoj za to
primerjavo; `reports/THUD_VMESNO/data-audit.json` določa isti podnabor 20 slik.
Anotacije se berejo šele pri vrednotenju in izrisu referenc za pregled.

Potrebujemo Docker z NVIDIA Container Toolkit, delujoč RTX 5090 in prostor za
približno 18 GB uteži ter dve PyTorch sliki. Gostiteljevega Pythona 3.14 in
globalnih paketov ne spreminjamo. Host uporablja obstoječe zahteve projekta;
native modela uporabljata Python 3.11, torch 2.7.1+cu128 in Transformers 4.46.0.
CPU nadomestilo je prepovedano in vsi parametri se preverijo na CUDA.

```bash
bash scripts/thud_extended.sh prepare
bash scripts/thud_extended.sh nvila 1
bash scripts/thud_extended.sh nvila 100
bash scripts/thud_extended.sh florence 1
bash scripts/thud_extended.sh florence 100
bash scripts/thud_extended.sh reason
bash scripts/thud_extended.sh report
```

`all` izvede faze zaporedno. `prepare` prenese javni, pripeti reviziji, pripravi
uradno VILA kodo in dve ločeni Docker sliki. GPU modela tečeta zaporedno.
Ob koncu native procesa se njegov GPU pomnilnik sprosti. `reason` uporablja
obstoječo projektno Ollamo in Qwen BF16; nobenega starega opisa ne generira znova.
`report` je v celoti brez inferenc. Ob procesni oviri NVILA `all` zapiše izhodno
kodo in nadaljuje Florence. Posamezno fazo lahko ponovno poženemo.

Izvajalnika že shranjene merjene odgovore, tudi zavrnitve, ohranita. Ob ponovnem
nalaganju fizično ogrejeta model in ogrevanja ločita od 100 merjenih položajev.
Izhodni manifest zavrne spremenjene nastavitve/vhodne slike. Nov eksperiment
zahteva novo mapo; datotek z napakami ne brišemo zaradi lepšega imenovalca.
Nedokončan klic po procesnem zrušenju nima izmišljenega odgovora/časa: procesna
napaka ostane ločena v setup/logs, nadaljevanje ponovi samo manjkajočo datoteko.

## NVILA

`thud_nvila.py` neposredno kliče uradni `llava.load`/`generate_content`, zato
ni odvisen od starega API seznama, ki zavrne pravilno ime NVILA-8B. Slikovni
API strežnik ni potreben; ne trdimo, da je stari strežnik popravljen. Lokalni
model ostaja točno `Efficient-Large-Model/NVILA-8B`.

Uradna koda je pripeta na `57ecbfb904b99b563db69ef7916f6390eecea516`;
`scripts/patch_nvila_inference.py` odstrani neuporabljene FP8/učne uvoze in
stari FlashAttention zamenja s SDPA. Dinamični S2, Qwen2, SigLIP, projektor in
uteži ostanejo. Patch se shrani v `setup/vila-inference.patch`.
Uradnega `pip install -e .` ne uporabimo, ker bi vsilil torch 2.3 in staro CUDA
okolje. Loader in slikovni tenzorji dejansko uporabljajo FP16, ne BF16.

NVILA prejme nespremenjeni `thud-description-rich.txt` in isti generični vhod
brez referenc kot 2. del. Greedy T=0, seed 42, 16384 kontekst in največ 8192
novih tokenov; top-p/top-k pri greedy ne vplivata. Kazen prisotnosti Ollama ni
podprta. `repetition_penalty=1.05` je vrednost iz uradnih uteži; začetni pilot
z ovojevo 1.0 in ponavljajočim JSON je ohranjen kot `diagnostic-nvila-rp1/`.
To je razlika nastavitev, ne popravljanje neuspelega odgovora.

## Florence

`thud_florence.py` uporablja točno `microsoft/Florence-2-large` (brez `-ft`) in
Microsoftov procesor/razčlenjevalnik. Vsaka slika dobi tri neodvisne naloge:
`<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`.
Qwenov JSON prompt se ne pošlje. Nastavitve sledijo uradnemu primeru:
FP16, num_beams=3, do_sample=false, max_new_tokens=1024, seed 42.
Shranjeni so surovi tokeni/ID-ji in izvirni razčlenjeni izhodi. Bbox so xyxy v
pikah izvirne slike; validator jih ne obrezuje ali preureja.

Ločeno merimo vsako nalogo in celoten kombinirani postopek treh nalog na sliko,
s sinhronizacijo CUDA. Kombinirani čas vsebuje predprocesiranje, generiranje,
razčlenjevanje in zapise JSON, izris pa je ločen. Nalaganje/ogrevanje nista del
glavnih časov. PyTorch allocated/reserved ni isto kot Ollama size_vram ali
skupni nvidia-smi. Stari časi vključujejo njihove dokumentirane omejitve.

## Besedilna pretvorba in kakovost

`thud_extended_reason.py` uporabi NVILA veljavni rich JSON nespremenjen.
Če rich izhod ni veljaven, ločeni `literal-native-text-v1` ohrani **celotno surovo
besedilo** kot sceno in iz prvega strogo razčlenjenega predponskega objekta JSON
vzame dobesedna imena ter celotne izvirne objekte kot opazovanja. Izvirna rich
zavrnitev ostane nespremenjena; to ni popravljen rich JSON ali dokaz pravilnosti.
Podvojeni ključi in manjkajoča imena so še vedno zavrnjeni. Nadaljnje besedilo
ni izbrisano; seznam transportnih objektov ne obljublja izčrpnosti vseh navedb
v njem. Tako razvrščanje uporablja tudi dejanske modelove trditve iz zavrnjenih
opisov brez dodajanja izmišljenih vsebin. Razlika proti stari strogi vstopni
shemi je izrecno označena; teh rezultatov ne razlagamo kot izboljšavo sheme.
Florencejeve oznake in fraze ter okvirje dobesedno zapiše v opazovanje.
Vsako regijo OD/dense obdrži posebej; ne izmisli ujemanja instanc, odnosov,
materialov, stikov ali namestitve. `not_visible` je transportna oznaka za
nepodan dokaz; `identifiable=true` pomeni, da je model poimenoval predlagano
regijo, ne ročne potrditve. Isti Qwen BF16 dobi samo zamrznjeno besedilo,
short-no-examples, T=0.3, seed 42, enake omejitve kot prej. Ne dobi slik.

`configs/thud-florence-matching.json` vnaprej določa slovar imen in IoU 0.5.
`thud_box_metrics.py` izvaja enolično največje dvodelno ujemanje po isti sliki
in razredu; pari in oba okvirja so shranjeni. Poročamo pokritost ustreznih
anotiranih regij, ne accuracy celotne scene ali pravilnosti fraz. Skupine in
razredi zunaj slovarja so izključeni; neujemanje ni avtomatično halucinacija.
Ni mAP brez ustreznih detekcijskih ocen; ni accuracy stabilnosti brez referenc.

## Izhodi

- `results/thud-extended/{nvila,florence}/`: manifesti, odgovori, okolja,
  ogrevanja, source snapshots; Florence tudi `overlays/` in `combined/`.
- `results/thud-extended/{models,vendor,setup,logs}/`: uteži, koda, popravki,
  konkretne napake in rešitve. Uteži so pripete na SHA revizije v downloaderju.
- `results/thud-extended/{frozen,reason}/`: izvorni opisi in nove presoje.
- `results/thud-extended/manual-review.html`: isti20, slike, reference, stare
  in nove napovedi ter razvrščanje. Preglednica pending ni ročno potrjena resnica.
- `reports/THUD_VMESNO/extended-*`: majhni agregati, metodologija, pari, napake
  in delne ročne opombe. `extended.tex` se vključi v celoten PDF/LaTeX.

Prejšnji odgovori imajo indeks SHA256; offline poročilo preveri ohranitev.
Velike surove datoteke in uteži ostanejo lokalno. Git vsebuje kodo in samostojen
paket poročila. Testi: `python3 -m unittest discover -s tests -v`.

Uradni viri: [NVILA uteži](https://huggingface.co/Efficient-Large-Model/NVILA-8B),
[VILA koda](https://github.com/NVlabs/VILA),
[Florence naloge in primeri](https://huggingface.co/microsoft/Florence-2-large),
[PyTorch CUDA različice](https://pytorch.org/get-started/previous-versions/).
