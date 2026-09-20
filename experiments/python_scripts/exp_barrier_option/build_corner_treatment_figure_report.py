r"""Down-and-out put: assemble the figures of a corner-treatment comparison into one PDF.

Collects, WITHOUT recomputing anything, the figures and Markdown tables already
written by

- ``aggregate_terminal_function_comparison.py`` (comparison figure, window-shape
  sweep, band network contribution, ``table.md``, ``s_band_errors.md``),
- ``evaluate_greeks_no_corner.py`` (Gamma-error figure, ``greeks_no_corner_table.md``),
- ``pilot_down_and_out_put.py`` (per-run price surface, log slices and, for the
  analytic corner treatments, the decomposition figure; one run per configuration,
  the seed given by ``--seed``),

copies them under ``<out_dir>/figures/``, writes a self-contained LaTeX file
(French, one figure per section with the caption stating what is plotted and
which artefact it comes from, the Markdown tables converted to ``tabular``) and
compiles it with ``latexmk -pdf`` when available. The report's run list comes
from the aggregation's ``summary.yaml`` so the figures and tables are those of
one and the same aggregation.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/build_corner_treatment_figure_report.py \
        --aggregation-dir data/aggregate_terminal_function_comparison/<dir> \
        --greeks-dir data/evaluate_greeks_no_corner/<dir> \
        --out-dir rapports/corner_treatments_<date>
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.utils.run_context import find_repo_root  # noqa: E402
from aggregate_terminal_function_comparison import CONFIGURATION_LABELS  # noqa: E402

logger = logging.getLogger("build_corner_treatment_figure_report")

# French, one-line description of every configuration key of the aggregation.
CONFIGURATION_DESCRIPTIONS_FR: dict[str, str] = {
    "raw": r"lissage du coin ($\zeta$, $\varepsilon$), payoff brut $(K-s)^+$",
    "smoothed": r"lissage du coin, payoff lissé de Chen--Mangasarian",
    "blackscholes": r"lissage du coin, profil Black--Scholes $V^e$, route autograd ordinaire",
    "blackscholes_analyticres": r"lissage du coin, profil Black--Scholes $V^e$, route analytique à deux termes",
    "split": r"lissage du coin, profil split-semigroupe",
    "subtraction_raw": r"soustraction exacte ($\Delta V_{DOD}$), profil payoff brut (contrôle négatif)",
    "subtraction_blackscholes": r"soustraction exacte ($\Delta V_{DOD}$), profil Black--Scholes $V^e$",
    "subtraction_split": r"soustraction exacte ($\Delta V_{DOD}$), profil split-semigroupe",
    "enrichment_raw": r"enrichissement du coin ($\chi\Delta\,\mathrm{erf}(\xi)$), profil payoff brut (contrôle négatif)",
    "enrichment_blackscholes": r"enrichissement du coin ($\chi\Delta\,\mathrm{erf}(\xi)$), profil Black--Scholes $V^e$",
    "enrichment_split": r"enrichissement du coin ($\chi\Delta\,\mathrm{erf}(\xi)$), profil split-semigroupe",
}


def latex_escape_text(text: str) -> str:
    """Escape LaTeX specials in a plain-text cell while leaving ``$...$`` math untouched."""
    pieces = re.split(r"(\$[^$]*\$)", text)
    out = []
    for piece in pieces:
        if piece.startswith("$") and piece.endswith("$"):
            out.append(piece)
        else:
            piece = piece.replace("\\", r"\textbackslash{}")
            for char in "&%#_{}":
                piece = piece.replace(char, "\\" + char)
            piece = piece.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
            out.append(piece)
    return "".join(out)


def markdown_tables(markdown_path: Path) -> list[tuple[str, list[list[str]]]]:
    """Every ``| ... |`` table of a Markdown file, with the nearest preceding heading."""
    tables: list[tuple[str, list[list[str]]]] = []
    heading = ""
    rows: list[list[str]] = []
    for line in markdown_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue  # separator row
            rows.append(cells)
        elif rows:
            tables.append((heading, rows))
            rows = []
    if rows:
        tables.append((heading, rows))
    return tables


def table_to_latex(rows: list[list[str]], caption: str, label: str, font_size: str = r"\scriptsize") -> str:
    header, body = rows[0], rows[1:]
    n_columns = len(header)
    column_spec = "l" + "c" * (n_columns - 1)
    lines = [
        r"\begin{table}[H]", r"\centering", font_size,
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule",
        " & ".join(r"\textbf{" + latex_escape_text(c) + "}" for c in header) + r" \\", r"\midrule",
    ]
    for row in body:
        row = row + [""] * (n_columns - len(row))
        lines.append(" & ".join(latex_escape_text(c) for c in row[:n_columns]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


def figure_block(relative_path: str, caption: str, label: str, width: str = r"\linewidth",
                 landscape: bool = False) -> str:
    """One figure environment; ``landscape`` puts it on its own rotated page
    (for the very wide multi-panel comparison figures)."""
    lines = [
        r"\begin{figure}[H]", r"\centering",
        rf"\includegraphics[width={width}]{{{relative_path}}}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{figure}", "",
    ]
    if landscape:
        lines = [r"\begin{landscape}", *lines, r"\end{landscape}", ""]
    return "\n".join(lines)


def path_block(path_text: str) -> str:
    """A file path that may break across lines (url package's \\path)."""
    return r"\path{" + path_text + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation-dir", type=str, required=True,
                        help="Output directory of aggregate_terminal_function_comparison.py.")
    parser.add_argument("--greeks-dir", type=str, default=None,
                        help="Output directory of evaluate_greeks_no_corner.py (optional).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed whose per-run figures (price surface, log slices, decomposition) are shown.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Report directory (default: rapports/corner_treatments_<timestamp>/).")
    parser.add_argument("--title", type=str, default="Down-and-out put --- traitements du coin : figures",
                        help="Report title.")
    parser.add_argument("--no-compile", action="store_true", help="Write the .tex only.")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    aggregation_dir = Path(args.aggregation_dir).resolve()
    greeks_dir = Path(args.greeks_dir).resolve() if args.greeks_dir else None
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        repo_root / "rapports" / f"corner_treatments_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "build_report.log")])
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Aggregation: {aggregation_dir}")
    logger.info(f"Greeks: {greeks_dir}")
    logger.info(f"Report directory: {out_dir}")

    with open(aggregation_dir / "summary.yaml") as f:
        summary = yaml.safe_load(f)
    iters = summary.get("iters")
    epsilon = summary.get("epsilon")
    # summary.yaml layout: {"iters", "epsilon", ..., "configurations": {key: {"label", "runs": {seed: dir}, ...}}}
    if "configurations" not in summary:
        raise SystemExit("summary.yaml has no 'configurations' block; aggregate first.")

    def copy_figure(source: Path, name: str) -> str | None:
        if not source.exists():
            logger.warning(f"  missing figure: {source}")
            return None
        target = figures_dir / name
        shutil.copy2(source, target)
        logger.info(f"  {source} -> {target}")
        return f"figures/{name}"

    sections: list[str] = []

    # ---- 1. across-seed comparison ---------------------------------------
    sections.append(r"\section{Comparaison entre configurations (médianes sur les graines)}")
    sections.append(
        "Chaque point est une graine maîtresse, le losange plein la médiane. Les configurations "
        "de lissage excluent la fenêtre de coin $N_{0.1}$ de la collocation ; les traitements "
        "analytiques (soustraction, enrichissement) incluent le coin, ce que l'étiquette de "
        "l'axe rappelle. Source : " + path_block(str(aggregation_dir.relative_to(repo_root))) + "."
    )
    rel = copy_figure(aggregation_dir / "figures" / "terminal_function_comparison.png", "terminal_function_comparison.png")
    if rel:
        sections.append(figure_block(
            rel, rf"Erreur relative $L^2$ hors fenêtre de coin (métrique de comparaison), sur tout le domaine, "
                 rf"dans la fenêtre, et meilleure perte intérieure ; {iters} itérations, $\varepsilon={epsilon:g}$ "
                 r"pour les runs de lissage.", "fig:comparison"))
    for heading, rows in markdown_tables(aggregation_dir / "table.md"):
        sections.append(table_to_latex(rows, "Métriques par configuration, médiane [min, max] sur les graines "
                                             r"(\texttt{table.md}).", "tab:metrics"))
        break

    # ---- 2. model-based diagnostics -------------------------------------
    diagnostics_dir = aggregation_dir / "model_based_diagnostics"
    if diagnostics_dir.exists():
        sections.append(r"\section{Diagnostics à partir des modèles sauvegardés}")
        rel = copy_figure(diagnostics_dir / "figures" / "rel_l2_vs_excluded_area_by_window_shape.png",
                          "rel_l2_vs_excluded_area_by_window_shape.png")
        if rel:
            sections.append(figure_block(
                rel, r"Erreur relative $L^2$ sur le complémentaire d'une fenêtre exclue autour du coin, pour trois "
                     r"formes de fenêtre (losange $\ell^1$, parabole, hyperbole), en fonction de la fraction d'aire "
                     r"exclue.", "fig:window-shapes"))
        rel = copy_figure(diagnostics_dir / "figures" / "band_network_contribution.png", "band_network_contribution.png")
        if rel:
            sections.append(figure_block(
                rel, r"Bande $0.1<|s-B|<0.3$ : normes $L^2$ de $\Phi_\theta - V_{DO}$ (solution entraînée) et de "
                     r"$g_2 - V_{DO}$ (extension seule, sans réseau).", "fig:band"))
        band_tables = markdown_tables(diagnostics_dir / "s_band_errors.md")
        for heading, rows in band_tables:
            if heading.startswith("Relative"):
                sections.append(table_to_latex(rows, r"Erreur relative $L^2$ par bande de $s$ (tout $t$, fenêtre "
                                                     r"de coin retirée), médiane [min, max] sur les graines.", "tab:bands-rel"))
            elif heading.startswith("Absolute"):
                sections.append(table_to_latex(rows, r"Erreur absolue $L^2$ par bande de $s$ (norme discrète pondérée "
                                                     r"par l'aire des cellules).", "tab:bands-abs"))

    # ---- 3. Greeks ---------------------------------------------------------
    if greeks_dir is not None and greeks_dir.exists():
        sections.append(r"\section{Grecques au strike}")
        sections.append("Source : " + path_block(str(greeks_dir.relative_to(repo_root))) + ". "
                        r"Référence : dérivée symbolique (sympy) de la forme fermée de Reiner--Rubinstein.")
        rel = copy_figure(greeks_dir / "figures" / "greeks_no_corner.png", "greeks_no_corner.png")
        if rel:
            sections.append(figure_block(
                rel, r"Erreur relative sur $\Gamma(K,t)=\partial_{ss}\Phi_\theta(K,t)$ en fonction de $\tau=T-t$, "
                     r"une courbe par configuration (médiane sur les graines ; points : graines).", "fig:greeks"))
        for heading, rows in markdown_tables(greeks_dir / "greeks_no_corner_table.md"):
            sections.append(table_to_latex(rows, r"$\Delta$ et $\Gamma$ au strike : erreur relative ponctuelle, médiane "
                                                 r"[min, max] sur les graines ; (*) marque un $\Gamma$ exact proche d'un "
                                                 r"changement de signe.", "tab:greeks", font_size=r"\tiny"))
            break

    # ---- 4. per-run figures ------------------------------------------------
    sections.append(rf"\section{{Figures par run (graine {args.seed})}}")
    for configuration, entry in summary["configurations"].items():
        runs = entry.get("runs", {})
        run_dir_text = runs.get(args.seed) or runs.get(str(args.seed))
        if run_dir_text is None:
            logger.warning(f"  {configuration}: no run for seed {args.seed}; skipped")
            continue
        run_dir = Path(run_dir_text)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
        run_figures = run_dir / "figures"
        if not run_figures.exists():
            logger.warning(f"  {configuration}: {run_figures} missing; skipped")
            continue
        label_text = CONFIGURATION_LABELS.get(configuration, configuration).replace("\n", " ")
        description = CONFIGURATION_DESCRIPTIONS_FR.get(configuration, "")
        sections.append(rf"\subsection{{{latex_escape_text(label_text)}}}")
        sections.append(description + r". Run : " + path_block(run_dir.name) + ".")
        for pattern, caption in (
            ("price_surface_eps*.png", r"Surface de prix : solution entraînée, forme fermée, différence."),
            ("log_slice_eps*.png", r"Coupes $V(s,t)$ à $t$ fixé, échelle logarithmique (solution entraînée en trait "
                                   r"plein, forme fermée en tirets)."),
            ("subtraction_decomposition.png", r"Décomposition de l'estimateur : partie singulière en forme fermée, "
                                              r"extension régulière $h$, réseau $g_1u_\theta$, total, contre $V_{DO}$."),
        ):
            for source in sorted(run_figures.glob(pattern)):
                rel = copy_figure(source, f"{configuration}_seed{args.seed}_{source.name}")
                if rel:
                    sections.append(figure_block(rel, caption + " " + description + ".",
                                                 f"fig:{configuration}-{source.stem}"))

    # ---- write and compile --------------------------------------------------
    tex = "\n".join([
        r"% Generated by build_corner_treatment_figure_report.py -- do not edit by hand; rerun the script.",
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=0.8in]{geometry}",
        r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage[french]{babel}",
        r"\usepackage{amsmath,amssymb}", r"\usepackage{booktabs}", r"\usepackage{graphicx}",
        r"\usepackage{adjustbox}", r"\usepackage{float}", r"\usepackage{xcolor}", r"\usepackage{pdflscape}",
        r"\usepackage{url}", r"\urlstyle{tt}",
        r"\usepackage[colorlinks=true,allcolors=blue!55!black]{hyperref}",
        rf"\title{{{args.title}}}", r"\author{}", rf"\date{{{datetime.now().strftime('%Y-%m-%d')}}}",
        r"\begin{document}", r"\maketitle", r"\tableofcontents", r"\clearpage",
        r"Recueil des figures et tables produites par les scripts d'agrégation, sans aucun recalcul : "
        r"chaque légende indique ce qui est tracé ; les chemins des artefacts sources sont donnés en tête "
        r"de section. Notation : $\Delta=K-B$, $V_{DOD}$ le prix du digital down-and-out, "
        r"$\xi=\ln(s/B)/(\sigma\sqrt{2(T-t)})$, $\chi$ le cutoff de l'enrichissement, "
        r"$N_{0.1}=\{|s-B|+(T-t)\le0.1\}$ la fenêtre de coin des métriques.",
        "", *sections, r"\end{document}", "",
    ])
    tex_path = out_dir / "corner_treatment_figures.tex"
    tex_path.write_text(tex)
    logger.info(f"LaTeX written -> {tex_path}")
    if args.no_compile:
        return
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        logger.warning("latexmk not found; compile the .tex manually.")
        return
    result = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                            cwd=out_dir, capture_output=True, text=True, errors="replace")
    (out_dir / "latexmk.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        logger.error(f"latexmk failed (exit {result.returncode}); see {out_dir / 'latexmk.log'}")
        sys.exit(1)
    subprocess.run([latexmk, "-c", tex_path.name], cwd=out_dir, capture_output=True, text=True, errors="replace")
    logger.info(f"PDF -> {out_dir / 'corner_treatment_figures.pdf'}")


if __name__ == "__main__":
    main()
