#!/usr/bin/env perl

# Project-local latexmk configuration
# - Local: final PDFs → documents/latex_output/pdf/, artifacts/logs → documents/latex_output/build/
# - Overleaf: conservative mode (no shell hooks, no file moves)

use File::Basename;

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';

# Resolve paths relative to the repository root (directory of this latexmkrc)
my $rootdir = dirname(__FILE__);
my $auxdir  = "$rootdir/documents/latex_output/build";
my $pdfdir  = "$rootdir/documents/latex_output/pdf";

my $is_overleaf_env = exists $ENV{'OVERLEAF'} && $ENV{'OVERLEAF'};
my $is_overleaf_jobname = defined $jobname && $jobname eq 'output';
my $is_overleaf_jobname_arg = join(' ', @ARGV) =~ /(?:^|\s)-jobname=output(?:\s|$)/;
my $is_overleaf = $is_overleaf_env || $is_overleaf_jobname || $is_overleaf_jobname_arg;

if ($is_overleaf) {
	# Overleaf compatibility mode: keep defaults and avoid external shell commands.
} else {
	# Put aux/log artifacts under documents/latex_output/build (absolute path)
	$emulate_aux = 1;
	$aux_dir = $auxdir;
	$out_dir = $auxdir;

	# On successful build: move only PDF, synctex, and tar.gz to pdf/; leave all other artefacts in build/.
	# Symlink build/paper.pdf -> ../pdf/paper.pdf so the editor viewer (which looks in outDir=build) still finds it.
	my $q = "'" . $pdfdir . "'";
	$success_cmd =
	  'if [ "$(basename "%D")" = "output.pdf" ]; then exit 0; fi; ' .
	  'mkdir -p ' . $q . ' && ' .
	  'mv -f "%D" ' . $q . '/ && ' .
	  'base="$(dirname "%D")/$(basename "%D" .pdf)"; ' .
	  'syn="${base}.synctex.gz"; tgz="${base}.tar.gz"; ' .
	  '[ -f "$syn" ] && mv -f "$syn" ' . $q . ' || true; ' .
	  '[ -f "$tgz" ] && mv -f "$tgz" ' . $q . ' || true; ' .
	  'ln -sf ../pdf/$(basename "%D") "$(dirname "%D")/$(basename "%D")"';
}

