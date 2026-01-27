# This script defines the default environment the pipeline runs in.
# Add the required packages to execute the code necessary for each derivation.
# If you want to create visual representations of the pipeline, consider adding
# `{visNetwork}` and `{ggdag}` to the list of R packages.
library(rix)

# Define execution environment

rix(
  date = "2026-01-19",
  r_pkgs = c(
    "dplyr",
    "ggplot2",
    "reticulate",
    "yardstick",
    "testthat",
    "withr"
  ),
  git_pkgs = list(
    list(
      package_name = "rix",
      repo_url = "https://github.com/ropensci/rix/",
      commit = "HEAD"
    ),
    list(
      package_name = "rixpress",
      repo_url = "https://github.com/ropensci/rixpress",
      commit = "HEAD"
    )
  ),
  py_conf = list(
    py_version = "3.13",
    py_pkgs = c(
      "numpy",
      "pandas",
      "scikit-learn",
      "matplotlib",
      "seaborn",
      "pytest"
    ),
    git_pkgs = list(
      list(
        package_name = "ryxpress",
        repo_url = "https://github.com/b-rodrigues/ryxpress",
        commit = "HEAD"
      )
    )
  ),
  ide = "none",
  project_path = ".",
  overwrite = TRUE
)
