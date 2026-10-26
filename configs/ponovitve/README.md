# Protokol ponovitev

`protocol.json` določa 420 VLM generacij ter 240 položajev LLM. Izbor je v
`results/ponovitve/frames.json`, objavljena relativna kopija v manifestu poročila.

- Qwen/Gemma: `description-v1.txt`, preprosta obstoječa legacy shema; T=.3,
  semena 42/43/44, največ 4096 tokenov, kontekst 16384.
- NVILA: vsebinsko primerljiv `nvila-description-v2.txt` z izrecno strukturo.
  Pilot v1 je vrnil prosti tekst; vsi njegovi odgovori ostanejo ločena diagnostika.
  To je razlika prompta, zato ne trdimo, da so razlike izoliran vpliv modela.
- Florence: native caption/OD/dense, beam=3, brez vzorčenja, 1024 tokenov.
  Tri sveže generacije so potrebne tudi ob dobesedno enakem odgovoru.
- Drugi korak: kopiji obstoječih kratkega/dolgega prompta brez primerov in
  `reasoning.txt`. Isti zamrznjeni Qwen BF16 r1 za oba modela in vsa semena.

Dejanske konfiguracije in polni poslanih prompti so dodatno shranjeni pri vsakem
odgovoru ter v `reports/PRIMERJAVA_PONOVITEV/prompts/`. Napake se ne popravljajo.
Za ponovitev uporabi `bash scripts/thud_ponovitve.sh all`; namestitve se ne ponavljajo.
