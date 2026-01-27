# tests/testthat/test-functions.R

library(testthat)
library(yardstick)
library(ggplot2)

# Load the functions under test (script-style project)
source("../../functions.R")

test_that("plot_confusion_matrix returns a ggplot object (happy path)", {
  df <- data.frame(
    truth = factor(c(0, 0, 1, 1), levels = c(0, 1)),
    estimate = factor(c(0, 1, 0, 1), levels = c(0, 1))
  )
  cm <- conf_mat(df, truth, estimate)

  p <- plot_confusion_matrix(cm, class_labels = c("Negative", "Positive"))

  expect_s3_class(p, "ggplot")
})

test_that("plot_confusion_matrix errors on non-2x2 confusion matrix (bad input)", {
  df <- data.frame(
    truth = factor(c("a", "b", "c", "a", "b", "c")),
    estimate = factor(c("a", "b", "c", "a", "b", "c"))
  )
  cm <- conf_mat(df, truth, estimate)

  expect_error(
    plot_confusion_matrix(cm, class_labels = c("A", "B")),
    "Expected a 2x2 confusion matrix"
  )
})

test_that("save_confusion_plot creates a real PNG file (happy path)", {
  df <- data.frame(
    truth = factor(c(0, 0, 1, 1), levels = c(0, 1)),
    estimate = factor(c(0, 1, 0, 1), levels = c(0, 1))
  )
  cm <- conf_mat(df, truth, estimate)

  withr::with_tempdir({
    out <- save_confusion_plot(cm, filename = "confusion_matrix.png")
    expect_true(file.exists(out))
    expect_gt(file.info(out)$size, 0)
  })
})

test_that("copy_file_r copies bytes and returns out_path (happy path)", {
  withr::with_tempdir({
    src <- "src.txt"
    dst <- "dst.txt"
    writeLines("hello", src)

    out <- copy_file_r(src, dst)

    expect_equal(out, dst)
    expect_true(file.exists(dst))
    expect_equal(readLines(dst), "hello")
  })
})

test_that("copy_file_r errors if source file does not exist (bad input)", {
  withr::with_tempdir({
    expect_error(
      copy_file_r("missing_file.png", "out.png"),
      "copy_file_r failed"
    )
  })
})
