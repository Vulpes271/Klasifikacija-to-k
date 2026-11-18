# Ločen dodatek: odometrijska premostitev

Glavno [poročilo](README.md) najprej prikazuje običajni AMCL brez dodatne premostitve. Ta dodatek primerja preskok LiDAR popravka in propagacijo iz zadnje zanesljive poze, kadar ostane manj kot 100 veljavnih zadetkov. Obe metodi sicer uporabljata kolesno odometrijo.

Pri pragu 100 zadetkov ni bilo naravnih vrzeli. Za naravno premostitev v sobi zato nimamo empiričnega rezultata. Umetni izpadi so vnaprej določeni pri 40, 90 in 180 s ter trajajo 0,5, 1 in 2 s. To niso selektivno izbrane samo uspešne vrzeli.

| Primer / način, umetne vrzeli | Položaj <10 cm [%] | Orientacija <4° [%] | Oba [%] |
|---|---:|---:|---:|
| reverse-original / standard | 33.93 ± 17.42 | 53.23 ± 20.41 | 32.12 ± 16.38 |
| reverse-original / bridge | 36.76 ± 17.79 | 54.75 ± 21.35 | 34.90 ± 16.73 |
| reverse-front / standard | 42.79 ± 13.01 | 64.09 ± 11.69 | 40.20 ± 12.13 |
| reverse-front / bridge | 43.82 ± 16.87 | 61.65 ± 20.23 | 41.38 ± 15.68 |
| reverse-three / standard | 34.92 ± 18.05 | 56.49 ± 18.29 | 33.13 ± 16.92 |
| reverse-three / bridge | 20.34 ± 7.74 | 36.40 ± 16.42 | 19.51 ± 7.12 |
| reverse-empty / standard | 93.50 ± 0.71 | 95.12 ± 0.57 | 89.55 ± 0.76 |
| reverse-empty / bridge | 93.13 ± 0.48 | 95.07 ± 0.64 | 89.22 ± 0.58 |
| forward-original / standard | 66.11 ± 0.83 | 75.47 ± 1.13 | 62.27 ± 0.91 |
| forward-original / bridge | 63.58 ± 7.24 | 73.58 ± 4.85 | 59.68 ± 7.00 |
| forward-front / standard | 66.44 ± 1.04 | 74.85 ± 1.01 | 62.80 ± 1.24 |
| forward-front / bridge | 61.76 ± 9.92 | 72.55 ± 4.17 | 58.04 ± 9.73 |
| forward-three / standard | 64.36 ± 7.36 | 77.40 ± 4.05 | 60.92 ± 7.27 |
| forward-three / bridge | 67.31 ± 1.26 | 77.90 ± 1.41 | 63.66 ± 1.50 |

Ta tabela ocenjuje celotno vožnjo, tudi posledice po vrzelih. [gaps-evaluated.json](data/gaps-evaluated.json) ločeno vsebuje samo napake med vrzelmi, največje uspešno trajanje/razdaljo, manjkajoči GT, manjkajoče ocene in čas do vsaj ene sekunde stabilnega skupnega uspeha. Nerazpoložljiv čas ponovne stabilizacije je neuspeh in ni zamenjan z ničlo.

Premostitev nima enotne prednosti v vseh primerih. Najdaljša preizkušena umetna vrzel je približno 2 s; ne sklepamo o daljših izpadih.
