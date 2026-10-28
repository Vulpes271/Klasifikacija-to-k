# Prostorska lokalizacija: 4 slike × 2 modela

Natanko osem novih generacij, brez ogrevalnih generacij, ponovitev, LLM ali PDF. Ista navodila zahtevajo originalne slikovne pike in objekte id/label/bbox. T=0.3, seed=42, top-p=0.8, top-k=20, repetition-penalty=1.05, max-new-tokens=4096. Obstoječa lokalna Qwen BF16 in NVILA FP16 okolja; Florence OD r1 samo kopiran.

`new_predictions/` vsebuje osem novih prikazov. Pri neveljavnem/praznem rezultatu je prikazan original z izrecnim napisom NO BOXES DRAWN; noben okvir ni izmišljen. `florence/` vsebuje štiri stare OD prikaze, `originals/` štiri originale. `responses/` ohrani celotno emitirano besedilo v .response.json (tudi neveljavno), raw metapodatke, čas/VRAM/tokene in sprejete rezultate. Koordinate niso ročno popravljene, preračunane ali omejene na rob slike. Celoten JSON v enem zaključenem markdown bloku se dobesedno prebere, brez spreminjanja njegove vsebine. Izvirni ovoj in formatna napaka ostaneta v raw/response datotekah. Ni kakovostnih metrik ali GT.

Odprite index.html. Ponovljiv ukaz: `bash scripts/bbox_comparison.sh`; shranjeni položaji se preskočijo, tudi če je izhod neveljaven.

## Dejanski izhodi

Qwen: en veljaven rezultat z okvirji, trije neveljavni (vključno z izhodom, ki je dosegel omejitev 4096 tokenov). NVILA: en veljaven rezultat z okvirji, dva prazna in en neveljaven z okvirjem zunaj slike. Ni dodatnih testov ali poskusov. Vseh osem prikazov je v new_predictions/.

Qwenov dnevnik vsebuje izpis SIGSEGV iz SDPA, vendar je isti izvajalnik nato shranil vse štiri odzive in se zaključil z 0. Vzrok izpisa ni dodatno raziskan. Nobena ponovitev ali eager generacija ni bila izvedena; oba modela uporabljata obstoječi SDPA okolji. Surovi dnevnik je ohranjen v provenance/.
