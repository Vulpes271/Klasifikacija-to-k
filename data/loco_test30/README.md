# LOCO: 30 slik za industrijski preizkus

30 izvornih JPEG slik iz petih okolij (6 na okolje), 1.305 oznak objektov.
Vir: [LOCO, tum-fml](https://github.com/tum-fml/loco), CC0-1.0; kopija licence je v `LICENSE`.
Slike so že v Gitu. Na Linuxu ni treba prenašati celotnega arhiva.

- `pregled.jpg`: pregled vseh slik.
- `discovery.jsonl`: slike brez referenčnih oznak; vhod za VLM.
- `objects.jsonl`: referenčne oznake z okvirji; samo za naknadno vrednotenje.
- `annotations.coco.json`: izvorne COCO oznake izbranih slik.
- `selection.json`, `source.json`, `SHA256SUMS.txt`: sledljivost izvora in izbora.

Razredi: `forklift`, `pallet_truck`, `pallet`, `small_load_carrier`, `stillage`.
Oznake niso popoln seznam vsega na sliki: stene, tla, regali in ljudje lahko ostanejo
neoznačeni. Baza nima referenčnih razredov fixed/movable/uncertain.

Izbor je določen pred izvajanjem modelov. Vsako okolje razdelimo na šest skupin
po numerično urejenih imenih datotek; v vsaki izberemo sliko z več različnimi
razredi, nato redkejšimi razredi in nato številom oznak (omejeno na 20 za izbor).
To je namenski zahtevnejši vzorec, ne naključna ali neodvisna testna množica.
Nekatere slike so zamegljene oziroma podobne. Imena datotek niso zanesljiv skupen
časovni potek med kamerami, zato skupine niso neodvisna časovna obdobja.

Pripeti izvorni COCO dokument dejansko vsebuje 5.097 slik in 151.428 oznak;
številke se razlikujejo od splošnega opisa na spletni strani. Tukaj uporabljamo
številke iz dokumenta. SHA in različica sta shranjena v `source.json`.

Ponovitev izbora v NOVO mapo (potrebuje Pillow, internet in približno 104 MB
izvornih oznak; iz ZIP arhiva prenese samo območja z izbranimi slikami):

```bash
python scripts/prepare_loco_sample.py --output data/loco_test30_recreated
```

Navodila za obdelavo: [CODEX_LOCO.md](../../CODEX_LOCO.md).
