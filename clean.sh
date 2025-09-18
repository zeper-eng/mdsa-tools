#!/bin/zsh

# Files and directories to remove from git cache
files_to_remove=(
  ".DS_Store"
  ".gitignore"
  "pyproject.toml"
  "MANIFEST.in"
  "README.md"
  "tests"
  "mdsa_tools.egg-info"
  "environments"
  "notebooks"
  "supplemental_figures"
  "resources"
)

# Remove files from git cache
for file in $files_to_remove; do
  git rm --cached $file
done

# Append files to .gitignore if not already present
for file in $files_to_remove; do
  if ! grep -Fxq "$file" .gitignore; then
    echo "$file" >> .gitignore
  fi
done

# Commit and push changes
git add .gitignore
git commit -m "Removed sensitive files from git cache and added to .gitignore"
git push

