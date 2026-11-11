# LILocBench: filtriranje med izdelavo zemljevida

**Celotni paket se še izvaja.** Spodaj so zaključeni referenčni zagoni na originalnem zemljevidu; rezultati filtriranega zemljevida še niso na voljo. To še ni primerjava učinka filtra. Končno poročilo na tej poti bo vsebovalo oba zemljevida, parne rezultate in trajektorije.

Zemljevida uporabljata samo `mapping`, isti javni GT na čas vsakega žarka in isto log-odds mrežo 5 cm. Odstranjeni žarki ne posodabljajo prostega prostora ali zadetka. Florence-2 large → lokalni Qwen3.5 27B Q4_K_M → SAM 2.1 Hiera Small; ~5 Hz, posamezna RGB oznaka, ±50 ms in obstoječa zaščita fixed/uncertain na prekrivanjih. Lokalizacijski skeni so originalni na obeh zemljevidih.

Kratek preizkus osmih parov uspe; uspe vseh 50 regresijskih testov. Načrt: 10 parov na vsaki od štirih sekvenc, semena 42–51, ista odometrija, inicializacija, AMCL parametri in seme znotraj para; brez poravnave.

## Zaključena referenca: originalni zemljevid, n=10

| Sekvenca | RMSE položaja [cm] | RMSE yaw [°] | Oba: <10 cm in <4° [%] |
|---|---:|---:|---:|
| static_0 | 6.32 ± 0.33 | 1.89 ± 0.05 | 89.40 ± 0.56 |
| dynamics_0 | 6.10 ± 0.21 | 1.55 ± 0.06 | 86.92 ± 1.77 |
| lt_changes_0 | 23.12 ± 2.51 | 7.39 ± 0.52 | 62.19 ± 1.16 |
| lt_changes_dynamics_0 | 12.26 ± 0.88 | 2.54 ± 0.12 | 79.85 ± 1.33 |

To ni bitno enaka reprodukcija članka: uradni zemljevid vključuje scan-matching/pose-graph izboljšavo mapping poz. Naša zemljevida uporabljata neposredni javni GT, oba enako; AMCL je obstoječi ROS2 Nav2 1.3.12 s pilotnimi parametri. Razliko filtra bomo merili s parno primerjavo naših dveh zemljevidov.

## Ponovitev / nadaljevanje

```bash
python3 -m llm_object_classifier.localization_filtered_map --config configs/lilocbench-filtered-map.json
```

Ukaz ponovno uporabi lokalne podatke in cache. Posamezni koraki: `--stage prepare`, `preflight`, `models`, `maps`, `amcl`, `report`. Napredek: `results/lilocbench-filtered-map/progress.json`, podrobnosti: `batch.log`. Surovi podatki, maske in galerija ostanejo lokalno. Končna galerija: `/home/student/jakob/Klasifikacija-to-k/results/lilocbench-filtered-map/gallery/index.html`.

[Vnaprej določen protokol](protocol.json), [projekcijski preizkus](preflight.json), [preizkus mreže](map-smoke.json).

Viri: [članek, tabela III](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/trekel2025iros.pdf), [uradni podatki in navodila](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/), [evaluator](https://github.com/PRBonn/LILocBench). Trekel et al., IROS 2025; podatki in izpeljani zemljevidi CC BY-NC-SA 4.0.
