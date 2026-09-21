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


def pivot_greeks_table(rows: list[list[str]], column: str) -> list[list[str]]:
    """Pivot the Greeks table (one row per configuration and t) into one row per
    configuration with one column per t, keeping the ``column`` cell
    (``err_rel_Delta`` or ``err_rel_Gamma``) as ``median [min, max]``."""
    header = rows[0]
    i_conf, i_t, i_value = header.index("Configuration"), header.index("t"), header.index(column)
    times: list[str] = []
    per_configuration: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        per_configuration.setdefault(row[i_conf], {})[row[i_t]] = row[i_value]
        if row[i_t] not in times:
            times.append(row[i_t])
    out = [["Configuration"] + [f"$t={t}$" for t in times]]
    for configuration, values in per_configuration.items():
        out.append([configuration] + [values.get(t, "—") for t in times])
    return out


def figure_block(relative_path: str, caption: str, label: str, width: str = r"\linewidth",
                 landscape: bool = False) -> str:
    """One figure environment; ``landscape`` puts it on its own rotated page
    (for the very wide multi-panel comparison figures)."""
    size = (r"width=\linewidth,height=0.8\textheight,keepaspectratio" if landscape else f"width={width}")
    lines = [
        r"\begin{figure}[H]", r"\centering",
        rf"\includegraphics[{size}]{{{relative_path}}}",
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
    parser.add_argument("--profiles-dir", type=str, default=None,
                        help="Output directory of compare_corner_treatments_profiles.py (optional).")
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
    profiles_dir = Path(args.profiles_dir).resolve() if args.profiles_dir else None
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
        r"Métriques, toutes calculées sur la grille d'évaluation $300\times100$ de "
        r"$\Omega=(B,s_\infty)\times(0,T)$ (pas $0.008$ en $s$, $0.01$ en $t$), avec "
        r"$N_w=\{(s,t):|s-B|+(T-t)\le w\}$ la fenêtre de coin, $w=0.1$ :"
        "\n"
        r"\[ \mathrm{rel}_{L^2}(A)=\frac{\|\Phi_\theta-V_{DO}\|_{L^2(A)}}{\|V_{DO}\|_{L^2(A)}}, "
        r"\qquad A\in\{\Omega\setminus N_{0.1}\ (\text{hors coin}),\ \Omega\ (\text{global}),\ N_{0.1}\ (\text{coin})\}, "
        r"\]"
        r"\[ \text{best loss}=\min_k\ \frac{1}{n_f}\sum_{(s,t)\in\text{batch}_k}\big(\mathcal L^{BS}\Phi_\theta(s,t)\big)^2, "
        r"\qquad \text{max\_abs}=\max_{A}|\Phi_\theta-V_{DO}|. \]"
        "\n"
        "Ces normes portent sur la \\emph{valeur} $\\Phi_\\theta-V_{DO}$, champ continu à dérivée bornée : "
        "un raffinement de la grille à $2400\\times800$ change chaque chiffre de moins de $1{,}3\\,\\%$ "
        "(vérifié sur trois runs, section 15.5 du document de méthodologie). "
        "Chaque point est une graine maîtresse, le losange plein la médiane sur les graines. "
        "\\textbf{Collocation} : les runs de lissage (section 11.1 de la méthodologie) excluent $N_{0.1}$ "
        "de la collocation ; les runs de soustraction et d'enrichissement incluent le coin, il n'y a rien "
        "à exclure --- l'étiquette de l'axe le rappelle. Source : "
        + path_block(str(aggregation_dir.relative_to(repo_root))) + "."
    )
    rel = copy_figure(aggregation_dir / "figures" / "terminal_function_comparison.png", "terminal_function_comparison.png")
    if rel:
        sections.append(figure_block(
            rel, rf"De gauche à droite, de haut en bas : $\mathrm{{rel}}_{{L^2}}(\Omega\setminus N_{{0.1}})$ (métrique de "
                 rf"comparaison), $\mathrm{{rel}}_{{L^2}}(\Omega)$, $\mathrm{{rel}}_{{L^2}}(N_{{0.1}})$ et la meilleure perte "
                 rf"intérieure ; {iters} itérations, $\varepsilon={epsilon:g}$ pour les runs de lissage, 5 graines "
                 r"par configuration.", "fig:comparison"))
    for heading, rows in markdown_tables(aggregation_dir / "table.md"):
        sections.append(table_to_latex(rows, r"$\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$, $\mathrm{rel}_{L^2}(\Omega)$, "
                                             r"$\mathrm{rel}_{L^2}(N_{0.1})$, $\max_{\Omega\setminus N_{0.1}}|\Phi_\theta-V_{DO}|$, "
                                             r"meilleure perte et itération correspondante ; médiane [min, max] sur les graines "
                                             r"(\texttt{table.md}).", "tab:metrics"))
        break

    # ---- 2. model-based diagnostics -------------------------------------
    diagnostics_dir = aggregation_dir / "model_based_diagnostics"
    if diagnostics_dir.exists():
        sections.append(r"\section{Diagnostics à partir des modèles sauvegardés}")
        sections.append(
            r"Mêmes modèles, même grille ; aucun réentraînement. Bandes en $s$ : "
            r"$\mathcal B=[s_1,s_2]\times(0,T)\setminus N_{0.1}$, "
            r"$\mathrm{rel}_{L^2}(\mathcal B)=\|\Phi_\theta-V_{DO}\|_{L^2(\mathcal B)}/\|V_{DO}\|_{L^2(\mathcal B)}$ et "
            r"$\mathrm{abs}_{L^2}(\mathcal B)=\|\Phi_\theta-V_{DO}\|_{L^2(\mathcal B)}$ avec "
            r"$\|f\|_{L^2(\mathcal B)}=(\sum_{\mathcal B}f^2\,\Delta s\,\Delta t)^{1/2}$. "
            r"La bande $[2,s_\infty]$ est à lire en absolu : $\|V_{DO}\|_{L^2}$ y vaut $1.1\times10^{-4}$, "
            r"une erreur relative de $3$ y correspond à une erreur absolue de $3\times10^{-4}$."
        )
        rel = copy_figure(diagnostics_dir / "figures" / "rel_l2_vs_excluded_area_by_window_shape.png",
                          "rel_l2_vs_excluded_area_by_window_shape.png")
        if rel:
            sections.append(figure_block(
                rel, r"$\mathrm{rel}_{L^2}(\Omega\setminus N)$ pour trois familles de fenêtre $N$ autour du coin "
                     r"(losange $N_w=\{|s-B|+\tau\le w\}$, parabole $N_c=\{|s-B|\le cB\sigma\sqrt\tau\}$, "
                     r"hyperbole $N_d=\{\tau(s-B)\le d\}$, $\tau=T-t$), en fonction de la fraction d'aire exclue "
                     r"$|N\cap\Omega|/|\Omega|$ ; un panneau par configuration.", "fig:window-shapes"))
        rel = copy_figure(diagnostics_dir / "figures" / "band_network_contribution.png", "band_network_contribution.png")
        if rel:
            sections.append(figure_block(
                rel, r"Bande $\mathcal B=\{0.1<|s-B|<0.3\}$ (tout $t$) : $\|\Phi_\theta-V_{DO}\|_{L^2(\mathcal B)}$ "
                     r"(losanges, solution entraînée) contre $\|g_2-V_{DO}\|_{L^2(\mathcal B)}$ (tirets rouges, "
                     r"extension seule, sans réseau) et $\|V_{DO}\|_{L^2(\mathcal B)}$ (pointillés, échelle). "
                     r"Un rapport proche de 1 signifie que le réseau n'apporte rien dans la bande.", "fig:band"))
        band_tables = markdown_tables(diagnostics_dir / "s_band_errors.md")
        for heading, rows in band_tables:
            if heading.startswith("Relative"):
                sections.append(table_to_latex(rows, r"$\mathrm{rel}_{L^2}(\mathcal B)$ par bande $\mathcal B$ de $s$ "
                                                     r"(tout $t$, $N_{0.1}$ retirée), médiane [min, max] sur les graines.", "tab:bands-rel"))
            elif heading.startswith("Absolute"):
                sections.append(table_to_latex(rows, r"$\mathrm{abs}_{L^2}(\mathcal B)=\|\Phi_\theta-V_{DO}\|_{L^2(\mathcal B)}$ "
                                                     r"par bande de $s$ (norme discrète pondérée par l'aire des cellules), "
                                                     r"médiane [min, max] sur les graines.", "tab:bands-abs"))

    # ---- 3. Greeks ---------------------------------------------------------
    if greeks_dir is not None and greeks_dir.exists():
        sections.append(r"\section{Grecques au strike}")
        sections.append(
            r"Erreur relative \emph{ponctuelle} en $s=K$, à cinq dates $t\in\{0,0.25,0.5,0.75,0.9\}$ :"
            r"\[ \mathrm{err}_{\mathrm{rel}}\,\Delta(t)=\frac{|\partial_s\Phi_\theta(K,t)-\partial_sV_{DO}(K,t)|}{|\partial_sV_{DO}(K,t)|},"
            r"\qquad \mathrm{err}_{\mathrm{rel}}\,\Gamma(t)=\frac{|\partial_{ss}\Phi_\theta(K,t)-\partial_{ss}V_{DO}(K,t)|}{|\partial_{ss}V_{DO}(K,t)|}. \]"
            r"Côté entraîné : $\partial_s$, $\partial_{ss}$ de $g_1u_\theta$ par deux passes autograd imbriquées "
            r"au point $(K,t)$, dérivées de $g_2$ en forme fermée (split, soustraction, enrichissement) ou par "
            r"autograd (lissage Black--Scholes) ; côté référence : dérivée symbolique (sympy, évaluation mpmath) "
            r"de la forme fermée de Reiner--Rubinstein. Aucune grille n'intervient : ce sont des valeurs "
            r"ponctuelles, pas des quadratures, donc la largeur du pic de $\partial_{ss}$ au strike n'est pas un "
            r"problème de résolution. La fragilité est celle d'un point : $\partial_{ss}V_{DO}(K,t)$ change de "
            r"signe vers $t\approx0.27$, et l'erreur relative est mal conditionnée près de ce zéro (lignes $t=0.25$). "
            "Médiane [min, max] sur 5 graines. Source : " + path_block(str(greeks_dir.relative_to(repo_root))) + "."
        )
        rel = copy_figure(greeks_dir / "figures" / "greeks_no_corner.png", "greeks_no_corner.png")
        if rel:
            sections.append(figure_block(
                rel, r"$\mathrm{err}_{\mathrm{rel}}\,\Gamma(t)$ en fonction de $\tau=T-t$ (log-log), une courbe par "
                     r"configuration (médiane sur les graines ; points pâles : graines). Les tirets verticaux marquent "
                     r"les $\tau$ où $|\partial_{ss}V_{DO}(K,t)|$ est inférieur à $10\,\%$ de son maximum (erreur "
                     r"relative mal conditionnée).", "fig:greeks"))
        for heading, rows in markdown_tables(greeks_dir / "greeks_no_corner_table.md"):
            sections.append(table_to_latex(pivot_greeks_table(rows, "err_rel_Delta"),
                                           r"$\mathrm{err}_{\mathrm{rel}}\,\Delta(t)$ au strike, une ligne par configuration, "
                                           r"une colonne par $t$ ; médiane [min, max] sur 5 graines.", "tab:greeks-delta"))
            sections.append(table_to_latex(pivot_greeks_table(rows, "err_rel_Gamma"),
                                           r"$\mathrm{err}_{\mathrm{rel}}\,\Gamma(t)$ au strike, idem. La colonne $t=0.25$ "
                                           r"est mal conditionnée ($\partial_{ss}V_{DO}(K,0.25)=-3.3\times10^{-2}$, proche de "
                                           r"son zéro) et n'est pas à lire comme une performance.", "tab:greeks-gamma"))
            break

    # ---- 4. profiles along s and Greeks against t (one seed) -----------------
    if profiles_dir is not None and profiles_dir.exists():
        sections.append(r"\section{Profils en $s$ et grecques en fonction de $t$ (graine " + str(args.seed) + ")}")
        sections.append(
            r"Une seule graine, pas de médiane : ces figures montrent \emph{où} l'erreur de chaque traitement se "
            r"trouve, ce que les métriques intégrées ne disent pas. Évaluation ponctuelle en float64 ; dérivées "
            r"comme à la section précédente (autograd imbriqué sur $g_1u_\theta$, forme fermée pour $g_2$). "
            "Source : " + path_block(str(profiles_dir.relative_to(repo_root))) + "."
        )
        rel = copy_figure(profiles_dir / "figures" / "profiles_price_delta_gamma.png", "profiles_price_delta_gamma.png")
        if rel:
            sections.append(figure_block(
                rel, r"Lignes : $\Phi_\theta(s,t)$, $\partial_s\Phi_\theta(s,t)$, $\partial_{ss}\Phi_\theta(s,t)$ ; "
                     r"colonnes : $t\in\{0,0.5,0.9,0.99\}$ ; forme fermée en tirets noirs. Prix en échelle "
                     r"linéaire ; $\Delta$ et $\Gamma$ en échelle symlog (linéaire sous $0.1$, logarithmique "
                     r"au-delà, des deux côtés de zéro).", "fig:profiles", landscape=True))
        rel = copy_figure(profiles_dir / "figures" / "absolute_errors_along_s.png", "absolute_errors_along_s.png")
        if rel:
            sections.append(figure_block(
                rel, r"Erreurs absolues ponctuelles le long de $s$ : $e_0=|\Phi_\theta-V_{DO}|$, "
                     r"$e_1=|\partial_s\Phi_\theta-\partial_sV_{DO}|$, $e_2=|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}|$ "
                     r"(échelle log). C'est la figure qui localise la valeur ajoutée des traitements analytiques : "
                     r"le lissage (bleu, rouge) est à $10^{-3}$--$10^{-2}$ partout et à $10^{-1}$ près du coin "
                     r"vers $t=T$ ; la soustraction (vert, orange) à $10^{-6}$--$10^{-5}$ ; l'enrichissement "
                     r"(cyan, rose) est entre les deux \emph{près du coin} ($s\in[0.6,1]$, bande de transition "
                     r"de $\chi$) et rejoint la soustraction au-delà de $s\approx1.2$. Les erreurs relatives par "
                     r"bande grandissent avec $s$ pour toutes les configurations parce que $V_{DO}\to0$, pas parce "
                     r"que l'erreur absolue grandit.", "fig:abs-errors", landscape=True))
        rel = copy_figure(profiles_dir / "figures" / "greeks_at_strike_vs_time.png", "greeks_at_strike_vs_time.png")
        if rel:
            sections.append(figure_block(
                rel, r"Haut : $\partial_s\Phi_\theta(K,t)$ et $\partial_{ss}\Phi_\theta(K,t)$ en fonction de $t$ "
                     r"(traits pleins), valeurs exactes en tirets. Bas : $\mathrm{err}_{\mathrm{rel}}\,\Delta(t)$ et "
                     r"$\mathrm{err}_{\mathrm{rel}}\,\Gamma(t)$ (échelle log). La verticale grise marque le zéro de "
                     r"$\partial_{ss}V_{DO}(K,\cdot)$.", "fig:greeks-vs-t", width=r"0.9\linewidth"))

    # ---- 5. per-run figures ------------------------------------------------
    sections.append(rf"\section{{Figures par run (graine {args.seed})}}")
    sections.append(
        r"Pour chaque configuration, le run de la graine " + str(args.seed) + r" : surface de prix sur la grille "
        r"d'évaluation (entraîné, forme fermée, différence signée) ; coupes $V(s,t)$ à $t$ fixé en échelle "
        r"symlog (linéaire sous $10^{-6}$, logarithmique au-delà) --- le prix va de $0.4$ près de la barrière à "
        r"$10^{-5}$ dans le champ lointain, invisible en linéaire, et un prix entraîné qui change de signe y "
        r"apparaît comme un passage sous zéro, pas comme une chute vers un plancher ; et, pour les traitements "
        r"analytiques, la décomposition de l'estimateur $\Phi_\theta=S+h+g_1u_\theta$ ($S=\Delta V_{DOD}$ ou "
        r"$S=\chi\Delta\,\mathrm{erf}(\xi)$, $h=\pi-\chi\,\pi(B,\cdot)$). Cette dernière montre ce que la partie "
        r"en forme fermée reproduit à elle seule, ce que l'extension régulière ajoute et ce qui reste au réseau : "
        r"pour la soustraction le réseau est presque nul (l'ansatz porte la solution) ; pour l'enrichissement "
        r"$E$ est localisé par $\chi$ et $h$ présente un creux compensateur dans la bande de transition "
        r"$[B+\delta_0,B+\delta_1]$, que le réseau doit corriger --- c'est là que se situe son surcroît d'erreur."
    )
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
            ("log_slice_eps*.png", r"Coupes $V(s,t)$ à $t\in\{0,0.5,0.9\}$, échelle symlog (solution entraînée en "
                                   r"trait plein, forme fermée en tirets ; trait vertical : $s=B$)."),
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
