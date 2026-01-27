# functions.R
# ===========
#
# R execution layer for the reproducible Nix + rixpress pipeline.
#
# This module provides:
# - Confusion-matrix visualization (ggplot heatmap) from a yardstick::conf_mat object
# - A plot-saving helper that produces a real PNG file (for pipeline artifacts)
# - An R-side file encoder used by rixpress to copy the PNG into the Nix derivation output
#
# Design goals:
# - Deterministic output
# - Explicit inputs/outputs
# - No reliance on interactive state
# - Works in sandboxed Nix builds
#
# Expected upstream objects:
# - confusion_matrix: a yardstick::conf_mat object with a 2x2 contingency table


library(ggplot2)
library(yardstick)

# ------------------------------------------------------------------------------
# plot_confusion_matrix()
# ------------------------------------------------------------------------------
# Purpose:
#   Create a "seaborn-like" confusion matrix heatmap as a ggplot object.
#
# Inputs:
#   confusion_matrix : yardstick::conf_mat
#       Confusion matrix object produced by yardstick::conf_mat().
#   class_labels : character vector of length 2 (default: c("Negative","Positive"))
#       Display labels for the two classes, used on x/y axes.
#
# Behavior:
#   - Extracts the underlying count table from confusion_matrix$table
#   - Validates it is exactly 2x2 (binary classification)
#   - Builds a tidy data.frame with (Actual, Predicted, Count)
#   - Returns a ggplot heatmap with tile coloring + numeric annotations
#
# Returns:
#   ggplot object (not printed automatically)
#
# Pipeline role:
#   Visualization step (R-side). Can be printed interactively or saved via ggsave().
plot_confusion_matrix <- function(confusion_matrix,
                                  class_labels = c("Negative", "Positive")) {
  m <- as.matrix(confusion_matrix$table)

  if (nrow(m) != 2 || ncol(m) != 2) {
    stop("Expected a 2x2 confusion matrix, got: ", nrow(m), "x", ncol(m))
  }

  df <- expand.grid(
    Actual = class_labels,
    Predicted = class_labels,
    KEEP.OUT.ATTRS = FALSE
  )

  # yardstick::conf_mat => rows = truth (Actual), cols = estimate (Predicted)
  # The order below maps:
  #   (Actual=Neg, Pred=Neg), (Actual=Neg, Pred=Pos),
  #   (Actual=Pos, Pred=Neg), (Actual=Pos, Pred=Pos)
  df$Count <- c(m[1, 1], m[1, 2], m[2, 1], m[2, 2])

  ggplot(df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile() +
    geom_text(aes(label = Count), color = "white", size = 5) +
    scale_fill_gradient(low = "#deebf7", high = "#08519c") +
    labs(
      title = "Confusion Matrix",
      x = "Predicted",
      y = "Actual"
    ) +
    theme_minimal()
}

# ------------------------------------------------------------------------------
# save_confusion_plot()
# ------------------------------------------------------------------------------
# Purpose:
#   Render and save the confusion matrix heatmap to a PNG file and return its path.
#
# Inputs:
#   confusion_matrix : yardstick::conf_mat
#       Confusion matrix to plot.
#   filename : str (default: "confusion_matrix.png")
#       Output file name/path. In Nix/rixpress builds, a flat filename is preferred.
#   width, height : numeric
#       Dimensions in inches passed to ggsave().
#   dpi : integer
#       DPI passed to ggsave().
#   class_labels : character(2)
#       Display labels for the axes.
#
# Behavior:
#   - Creates the ggplot via plot_confusion_matrix()
#   - Saves to PNG via ggsave(device="png")
#   - Returns the filename (path)
#
# Returns:
#   character(1) : path to the saved PNG
#
# Pipeline role:
#   Produces a real file for downstream artifact handling (copy into derivation output).
save_confusion_plot <- function(confusion_matrix,
                                filename = "confusion_matrix.png",
                                width = 6, height = 4, dpi = 150,
                                class_labels = c("Negative", "Positive")) {
  p <- plot_confusion_matrix(confusion_matrix, class_labels = class_labels)
  ggsave(filename, plot = p, width = width, height = height, dpi = dpi, device = "png")
  filename
}

# ------------------------------------------------------------------------------
# copy_file_r()
# ------------------------------------------------------------------------------
# Purpose:
#   rixpress R-side encoder that copies a file into the Nix derivation output path.
#
# Inputs:
#   src_path : character(1)
#       Path to an existing file produced during build (e.g., "confusion_matrix.png").
#   out_path : character(1)
#       The target output path provided by rixpress/Nix for the derivation artifact.
#
# Behavior:
#   - Copies src_path -> out_path (overwrite allowed)
#   - Stops with a clear error message if the copy fails
#
# Returns:
#   character(1) : out_path
#
# Pipeline role:
#   Ensures the artifact stored in Nix is the actual PNG bytes (not an R object).
copy_file_r <- function(src_path, out_path) {
  ok <- file.copy(src_path, out_path, overwrite = TRUE)
  if (!ok) stop("copy_file_r failed: ", src_path, " -> ", out_path)
  out_path
}

