library(rixpress)
library(dplyr)
library(yardstick)
library(ggplot2)

pipeline_list <- list(
  # 1) Ingest
  rxp_py_file(
    name = raw_df,
    path = "data/HeartDiseaseTrain-Test.csv",
    read_function = "pd.read_csv"
  ),

  # 2) Encode categoricals
  rxp_py(
    name = encoded_df,
    expr = "encode_categoricals(raw_df)",
    user_functions = "functions.py"
  ),

  # 3) Target distribution plot (artifact)
  rxp_py(
    name = target_dist_plot_png,
    expr = "make_target_dist_png(encoded_df)",
    user_functions = "functions.py",
    encoder = "copy_file"
  ),

  # 4) Correlation heatmap (artifact)
  rxp_py(
    name = correlation_heatmap_png,
    expr = "make_correlation_heatmap_png(encoded_df)",
    user_functions = "functions.py",
    encoder = "copy_file"
  ),

  # 5) Preprocess
  rxp_py(
    name = processed_data,
    expr = "make_processed_data(encoded_df)",
    user_functions = "functions.py"
  ),

  # 6) Train
  rxp_py(
    name = svm_rbf_model,
    expr = "train_svm_rbf(processed_data)",
    user_functions = "functions.py"
  ),

  # 7) Predict
  rxp_py(
    name = y_pred,
    expr = "predict_labels(svm_rbf_model, processed_data)",
    user_functions = "functions.py"
  ),
  
  # 8) Accuracy
  rxp_py(
    name = accuracy,
    expr = "compute_accuracy(processed_data, y_pred)",
    user_functions = "functions.py"
  ),

  # 9) Evaluation DF
  rxp_py(
    name = evaluation_df,
    expr = "make_evaluation_df(processed_data, y_pred)",
    user_functions = "functions.py"
  ),

  # 10) Export evaluation for R ingestion
  rxp_py(
    name = evaluation_csv,
    expr = "evaluation_df",
    user_functions = "functions.py",
    encoder = "write_to_csv"
  ),

  # 11) Load CSV back into R as a data.frame
  rxp_r(
    name = evaluation_df_r,
    expr = evaluation_csv,
    decoder = "read.csv"
    ),

  # 12) R: factorize
  rxp_r(
    name = eval_factors,
    expr = mutate(evaluation_csv, across(everything(), factor)),
    decoder = "read.csv"
  ),

  # 13) R: confusion matrix
  rxp_r(
    name = confusion_matrix,
    expr = conf_mat(eval_factors, truth, estimate)
  ),

  # 14) R: confusion matrix plot artifact
  rxp_r(
    name = confusion_matrix_plot_png,
    expr = save_confusion_plot(confusion_matrix),
    user_functions = "functions.R",
    encoder = "copy_file_r"
  )
)

pipeline_list |> rxp_populate(build = FALSE)

add_import("import pandas as pd", "default.nix")
add_import("import os", "default.nix")

rxp_dag_for_ci()
rxp_make()

# =========================================================
# Manual visualization helpers (NO copying, NO folders)
# =========================================================

get_artifact_file <- function(derivation_name) {
  # Try rxp_read first
  p <- rxp_read(derivation_name)

  # If it's already a real file path, use it
  if (file.exists(p)) return(p)

  # Otherwise resolve relative path against the derivation store root
  insp <- rxp_inspect()
  root <- insp$path[insp$derivation == derivation_name][1]
  p2 <- file.path(root, p)
  if (file.exists(p2)) return(p2)

  # If still not found, list files inside the derivation output and pick the first file
  all <- list.files(root, full.names = TRUE, recursive = TRUE)
  all <- all[file.info(all)$isdir == FALSE]
  if (length(all) == 0) stop("No files found in derivation output for: ", derivation_name)
  all[1]
}

show_png_derivation <- function(derivation_name) {
  library(png)
  library(grid)

  src <- get_artifact_file(derivation_name)

  # If readPNG doesn't like missing extension, copy to temp .png
  tmp <- tempfile(fileext = ".png")
  file.copy(src, tmp, overwrite = TRUE)

  img <- readPNG(tmp)
  grid::grid.raster(img)

  invisible(src)
}

# After running source("gen-pipeline.R"), you can do:
# show_png_derivation("target_dist_plot_png")
# show_png_derivation("correlation_heatmap_png")
# show_png_derivation("confusion_matrix_plot_png")

