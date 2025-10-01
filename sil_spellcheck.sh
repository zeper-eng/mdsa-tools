#!/usr/bin/env bash
set -euo pipefail

# fix_silhouette_typo.sh
# Usage:
#   ./fix_silhouette_typo.sh                 # fixes mdsa_tools only (default)
#   ./fix_silhouette_typo.sh .               # fixes entire repo
#   ./fix_silhouette_typo.sh mdsa_tools supplemental_figures

# --- config ---
# File globs to edit in-place (text/code only; notebooks skipped by default)
EDIT_EXTS=("*.py" "*.md" "*.rst" "*.txt" "*.toml" "*.yml" "*.yaml" "*.cfg" "*.ini")

# Misspelling variants -> correct spellings (content)
# Also rewrites function names that used the misspelling.
perl_rewrite='
  s/plot_sillohuette_scores/plot_silhouette_scores/g;
  s/plot_sillohette_scores/plot_silhouette_scores/g;
  s/plot_sillouhette_scores/plot_silhouette_scores/g;

  s/sillohuette/silhouette/g;
  s/sillohette/silhouette/g;
  s/sillouhette/silhouette/g;
  s/Sillohuette/Silhouette/g;
  s/Sillohette/Silhouette/g;
  s/Sillouhette/Silhouette/g;
'

# What to search for when renaming files
NAME_PATTERNS=('*sillohuette*' '*sillohette*' '*sillouhette*')

# --- args ---
if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("mdsa_tools")
fi

echo "Targets: ${TARGETS[*]}"
command -v perl >/dev/null 2>&1 || { echo "perl is required"; exit 1; }

# --- 1) Edit file contents in place ---
echo "==> Rewriting content typos in: ${EDIT_EXTS[*]}"
for root in "${TARGETS[@]}"; do
  [ -d "$root" ] || continue
  while IFS= read -r -d '' file; do
    # Use perl for portable in-place edits (handles macOS/BSD vs GNU)
    perl -0777 -i -pe "$perl_rewrite" "$file"
  done < <(
    # build a find expression covering our extensions
    find "$root" -type f \( $(printf -- '-name %q -o ' "${EDIT_EXTS[@]}") -false \) \
      -not -path '*/.git/*' \
      -not -path '*/dist/*' \
      -not -path '*/build/*' \
      -not -path '*/.eggs/*' \
      -not -path '*/.venv/*' \
      -not -path '*/venv/*' \
      -print0
  )
done

# --- 2) Rename files whose names contain the misspelling ---
echo "==> Renaming files with misspellings in their names"
for root in "${TARGETS[@]}"; do
  [ -d "$root" ] || continue
  while IFS= read -r -d '' f; do
    new="$f"
    new="${new//sillohuette/silhouette}"
    new="${new//sillohette/silhouette}"
    new="${new//sillouhette/silhouette}"
    if [[ "$f" != "$new" ]]; then
      echo "mv -v -- '$f' '$new'"
      mkdir -p "$(dirname "$new")"
      mv -v -- "$f" "$new"
    fi
  done < <(
    # build a find expression for name patterns
    find "$root" -type f \( $(printf -- '-name %q -o ' "${NAME_PATTERNS[@]}") -false \) \
      -not -path '*/.git/*' -print0
  )
done

echo "==> Done. Consider running your tests/plots to verify."

