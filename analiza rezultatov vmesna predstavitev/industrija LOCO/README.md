# LOCO: celotni shranjeni rezultati

Zaključena je že začeta serija na vseh 30 slikah. Po zadnjem navodilu uporabnika
ni dodatnih testov, ponovitev, obsežnih analiz ali novih poročil. Zaključni
pregled potrjuje samo obstoj zahtevanih datotek (`file-existence.json`).

- `01_vlm/`: vseh 450 celotnih native odgovorov, dejanski prompti, nastavitve,
  revizije, tokeni, čas, GPU/VRAM in napake. NVILA besedilo je ohranjeno tudi
  ob neveljavnem JSON. Florence caption/OD/dense ostajajo native naloge.
- Originali so v `data/loco_test30/images/` v korenu repozitorija: vseh 30 je
  že v Gitu. `01_vlm/<slika>/florence/r1–r3/*.predictions.jpg` vsebuje 180
  slik z izrisanimi napovedmi OD/dense. NVILA in Qwen nimata izmišljenih
  okvirjev. Referenčne oznake ostajajo ločeno v `data/loco_test30/`.
- `02_llm/frozen/`: vseh 270 zamrznjenih vhodov s SHA256 in lokalnimi ID-ji.
- `02_llm/responses/`: 810 popolnih svežnjev, po šest položajev za posamezno
  kombinacijo slika/VLM/ponovitev/LLM: oba prompta × tri LLM ponovitve.
  Skupaj 4860 položajev; ohranjeni so celotni odgovori, napake in preskoki.
  Natančni skupni prompti so v `prompts/` in `user-prompts/`; po SHA se brez
  izgube obnovijo dejanski zahtevki. Veliki podvojeni delovni izhodi ostanejo
  lokalno v `results/loco/`, uteži in okolja niso objavljeni.
- `02_llm/provenance/`: nastavitve, okolja, dnevniki in prekinitve. Šest
  Gemma odgovorov brez potrditve GPU je ohranjenih kot napake. Preklicani
  Gemma long/r2 vhod 36 nima shranjenega odgovora: izrecno je rekonstruiran
  samo deterministični zahtevek, klic ni ponovljen. Za vhod 69 je ohranjen
  izvirni odgovor; njegov odjemalski čas vključuje operativni premor.
  Prednaložitev modela je bila brez prompta/besedilne generacije.

Odprite `index.html` lokalno v celotnem prenesenem repozitoriju. Obstoječi
pregledovalnik prikazuje celotne VLM/LLM odgovore. Stari agregati, razlike,
obrazci in opombe so ohranjeni kot predhodni vmesni posnetki in niso na novo
izračunani; ne predstavljajo potrjene končne analize. Že prej izdelani
`Priloga_LOCO.pdf/.tex` sta ohranjeni. Dodatnega glavnega poročila ni.

Veljaven zapis in strinjanje nista pravilnost. Referenca ima pet logističnih
razredov, ni popoln seznam scene in nima fixed/movable/uncertain ground truth.
Prejšnji THUD poskusi niso ponovno izvedeni. Vsi nadaljnji ukazi za inferenco,
teste ali poročila so za to oddajo ustavljeni.
