"""Offline HTML comparison of photographs and actual saved descriptions."""
import base64
import html
import json
from pathlib import Path


def write_report(output, frames, results):
    escape = lambda value: html.escape(str(value))
    by_frame = {}
    for row in results:
        by_frame.setdefault(row["frame_id"], []).append(row)
    parts = ['''<!doctype html><html lang="sl"><meta charset="utf-8"><title>Primerjava opisov slik</title>
<style>body{font:16px system-ui;margin:32px;color:#17243b;background:#f4f6fa}main{max-width:1500px;margin:auto}
section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{max-width:640px;width:100%;height:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{border:1px solid #d3dce8;padding:18px;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #d3dce8;vertical-align:top}
small{color:#53627b}.error{color:#b22}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><main>
<h1>1. korak: primerjava opisov slik</h1><p>To so dejanski izhodi modelov, ne preverjene oznake.
Brez referenc ni meritve pravilnosti. Pri slikah brez oznak so tudi imena in število objektov predlog modela.
ID-jev različnih modelov pri takih slikah ne primerjajte kot iste objekte.</p>''']
    for frame in frames:
        fid = frame["record"]["frame_id"]
        parts.append(f'<section><h2>{escape(fid)}</h2><img alt="Izvirna fotografija" src="data:{frame["mime"]};base64,{base64.b64encode(frame["image"]).decode()}">')
        parts.append('<div class="grid">')
        for result in by_frame.get(fid, []):
            parts.append(f'<article><h3>{escape(result["model_name"])} / {escape(result["variant"]["id"])}</h3>')
            seconds = result.get("elapsed_seconds")
            time_text = "ni izmerjeno" if seconds is None else f"{seconds:.2f} s"
            parts.append(f'<small>Čas zahteve: {time_text} | GPU potrjen prek API: {result.get("gpu_verified", False)}</small>')
            if result["status"] != "ok":
                parts.append(f'<p class="error">{escape(result.get("error"))}</p><pre>{escape(result.get("content", ""))}</pre>')
            else:
                parts.append(f'<p>{escape(result["parsed"]["scene_description"])}</p><table><tr><th>Objekt</th><th>Opis</th><th>Dokazi</th></tr>')
                for obj in result["parsed"]["objects"]:
                    parts.append(f'<tr><td>{escape(obj["label"])} [{escape(obj["id"])}]</td><td>{escape(obj["observation"])}</td><td>{escape(obj["attachment_evidence"])}<br>identifiable={obj["identifiable"]}</td></tr>')
                    if result["parsed"].get("schema_version") == "rich-description-v2":
                        evidence = {k: obj[k] for k in ("appearance", "image_location", "occlusion", "support_contacts", "installation_evidence")}
                        parts.append('<tr><td colspan="3"><pre>' + escape(json.dumps(evidence, ensure_ascii=False, indent=2)) + '</pre></td></tr>')
                parts.append('</table>')
                if "relations" in result["parsed"]:
                    parts.append('<h4>Odnosi med lokalnimi ID-ji (trditve modela)</h4><pre>' + escape(json.dumps(result["parsed"]["relations"], ensure_ascii=False, indent=2)) + '</pre>')
            if result.get("unsupported_options"):
                parts.append('<p class="error">Parametri, ki niso poslani temu API-ju: ' + escape(', '.join(result['unsupported_options'])) + '</p>')
            parts.append('<details><summary>Nastavitve zahteve</summary><pre>' + escape(json.dumps(result.get('requested_options', {}), indent=2)) + '</pre></details></article>')
        parts.append('</div></section>')
    parts.append('</main></html>')
    Path(output).write_text('\n'.join(parts), encoding='utf-8')
