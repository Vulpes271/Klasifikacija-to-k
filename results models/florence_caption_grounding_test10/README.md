# Florence: detailed caption → phrase grounding, 10 slik

Nastavitve: `configs/stage1/florence-caption-grounding.json`.
MORE_DETAILED_CAPTION → CAPTION_TO_PHRASE_GROUNDING dobi natanko celoten novi
opis. Prej je bil opis že podroben, OD/dense pa sta delovala ločeno od njega.
Limit izhoda je 2048 namesto 1024; tri žarke, deterministično, FP16/eager,
izvirna obdelava slike iz checkpointa. Koordinate so v originalnih pikah.

Ukaz: `bash scripts/stage1_florence.sh`. Obstoječih deset LOCO slik, dve na
okolje. Dvajset novih native generacij, brez ponovitev, ogrevalnih generacij,
LLM ali PDF. Stari poskusi se ne spreminjajo. Nadaljevanje preskoči shranjeno.

Odprite index.html. Vsaka mapa: original, celoten opis in raw odgovora,
tokeni/nastavitve/čas/VRAM, novi okvirji in prejšnji OD za primerjavo.
`stage1/` vsebuje literalni transport za poznejši drugi korak, ki NI izveden.

Opis lahko izpusti predmete. Regije lahko opisujejo skupine ali se prekrivajo;
njihovo število ni število potrjenih fizičnih objektov ali kakovostna metrika.
Ni ročnega dodajanja/popravljanja okvirjev ali uporabe referenčnih oznak.

Uradna naloga: https://huggingface.co/microsoft/Florence-2-large#caption-to-phrase-grounding
