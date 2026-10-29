from pathlib import Path
import json

root = Path("porocila/LOCO_VLM_LLM")
source = Path("analiza rezultatov vmesna predstavitev/industrija LOCO/02_llm/responses")
tex = (root / "porocilo.tex").read_text(encoding="utf-8").replace("Delovno poročilo -- LOCO", "")
lines = [r"\clearpage", r"\subsection{Ostali LLM modeli}", "Isti vhod Qwen3-VL r1; temperatura 0.3, seme 42, ponovitev r1. Izvirni odgovori so brez popravkov."]
for frame in ("loco-s3-36540", "loco-s4-31835"):
    lines.append(r"\subsubsection{" + frame + "}")
    for model, title in (("qwen35-q4", "Qwen3.5 Q4"), ("gemma3-q4", "Gemma 3 Q4")):
        p = source / frame / "qwen3-vl" / "vlm-r1" / model / "responses.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for prompt, name in (("short", "Kratki"), ("long", "Dolgi")):
            rec = d[prompt]["r1"]
            lines += [r"\paragraph{" + title + " -- " + name + " poziv}",
                      "Status: " + rec["status"] + ". " + str(rec.get("error") or ""),
                      r"\begin{lstlisting}", rec["response"]["message"]["content"],
                      r"\end{lstlisting}"]
    if frame == "loco-s3-36540":
        lines.append("Komentar: Gemma spremeni klasifikacijo regalov; Qwen Q4 vrne nepravilno polje pri objektu 4.")
    else:
        lines.append("Komentar: Razlike v klasifikaciji rumenih zascit in betonskih tal.")
lines += [
    r"\\clearpage",
    r"\\section{Primerjava rezultatov}",
    "V tabelah je locena tehnicna veljavnost izhoda od pravilnosti zaznavanja in klasifikacije. Brez rocno potrjenega ground truth ni mogoce izracunati zanesljive tocnosti.",
    r"\\subsection{VLM}",
    r"\\begin{tabular}{|p{3.2cm}|p{3.3cm}|p{7.0cm}|}",
    r"\\hline Model & Strukturirani izhod & Opazanje \\\\ \\hline",
    r"Florence-2-large & 90/90 OD odzivov & Najbolj zanesljiv izris koordinat v dosedanjem postopku; pomembne objekte vcasih izpusti. \\\\ \\hline",
    r"Qwen3-VL & 90/90 opisov & Najbolj uporaben podroben opis scene med pregledanimi primeri; locena bbox naloga je uspela le pri eni od stirih slik. \\\\ \\hline",
    r"NVILA-8B & 0/90 popolnoma veljavnih JSON opisov & Besedilo je pogosto uporabno, izluscenje pa nezanesljivo. Bbox: en veljaven rezultat od stirih. \\\\ \\hline",
    r"\\end{tabular}",
    "Primerjava koordinat Qwen in NVILA temelji na samo stirih slikah, zato iz nje ne sklepamo o splosni natancnosti lokalizacije. Veljavnost Florence je tehnicna veljavnost OD, ne pravilnost okvirjev.",
    r"\\subsection{LLM}",
    "Spodaj je delez shematsko veljavnih odgovorov med dejansko izvedenimi klici v dosedanjem agregiranem pregledu (skupaj vec slik, VLM vhodov in ponovitev). Preskoceni vhodi niso vsteti. Veljavnost ni enaka pravilni klasifikaciji.",
    r"\\begin{tabular}{|p{4.5cm}|p{4cm}|p{4cm}|}",
    r"\\hline Model & Kratek poziv & Dolg poziv \\\\ \\hline",
    r"Qwen3.5 4B BF16 & 659/756 (87,2\\%) & 677/756 (89,6\\%) \\\\ \\hline",
    r"Qwen3.5 4B Q4 & 681/756 (90,1\\%) & 707/756 (93,5\\%) \\\\ \\hline",
    r"Gemma 3 4B Q4 & -- & -- \\\\ \\hline",
    r"\\end{tabular}",
    "Za Gemmo zakljucni agregat ni bil ponovno izracunan; stare vmesne stevilke zaradi nepopolnega obsega niso primerljive in jih zato ne navajamo.",
    "Po veljavnosti formata med primerljivima Qwen razlicicama vodi Qwen3.5 Q4 pri kratkem in dolgem pozivu. Dolgi poziv je pri obeh Qwen razlicicah zanesljivejsi. Gemma pri pregledanih primerih vcasih pravilneje klasificira regale, a lahko napacno oznaci betonska tla; najboljsi model po dejanski pravilnosti zato se ni dolocen.",
]
(root / "ostali_llm.tex").write_text("\n".join(lines), encoding="utf-8")
marker = r"\end{document}"
assert tex.count(marker) == 1
(root / "porocilo_celotno.tex").write_text(tex.replace(marker, r"\input{ostali_llm.tex}" + "\n" + marker), encoding="utf-8")
