from pathlib import Path
import json

root = Path("porocila/LOCO_VLM_LLM")
source = Path("analiza rezultatov vmesna predstavitev/industrija LOCO/02_llm/responses")
tex = (root / "porocilo.tex").read_text(encoding="utf-8")
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
(root / "ostali_llm.tex").write_text("\n".join(lines), encoding="utf-8")
marker = r"\end{document}"
assert tex.count(marker) == 1
(root / "porocilo_celotno.tex").write_text(tex.replace(marker, r"\input{ostali_llm.tex}" + "\n" + marker), encoding="utf-8")
