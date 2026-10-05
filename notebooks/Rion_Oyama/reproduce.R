# Replay the archived R Markdown without changing its model code.
# Usage: Rscript notebooks/Rion_Oyama/reproduce.R OUTPUT_DIR [INPUT_CSV]
argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) < 1L) stop('Supply a fresh output directory.')
script_arg <- grep('^--file=', commandArgs(), value = TRUE)
package_dir <- dirname(normalizePath(sub('^--file=', '', script_arg[1])))
source_file <- file.path(package_dir, 'executed_code', 'Project_codeW1.Rmd')
for (p in c('ranger', 'lightgbm', 'jsonlite', 'digest')) {
  if (!requireNamespace(p, quietly = TRUE)) stop('Install R package: ', p)
}
input <- if (length(argv) >= 2L) argv[2] else file.path(package_dir, '..', '..', 'data', 'train_predict_cost.csv')
input <- normalizePath(input, mustWork = TRUE)
expected <- 'dc851cc3c6e23d117b5018b4f133459945476fabfb44c9fc7380ba08db090c9d'
stopifnot(identical(digest::digest(file=input, algo='sha256'), expected))
out <- path.expand(argv[1])
if (dir.exists(out) && length(list.files(out, all.files=TRUE, no..=TRUE))) stop('Output directory must be empty.')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
out <- normalizePath(out)
dir.create(file.path(out, 'results'))
dir.create(file.path(out, 'results', 'per_fold'))
old <- setwd(file.path(out, 'results', 'per_fold'))
env <- new.env(parent=globalenv())
env$read.csv <- function(file, ...) {
  if (is.character(file) && length(file)==1L && grepl('^https://raw.githubusercontent.com/reinesana/STAT440-star-kingdom-project/', file)) file <- input
  utils::read.csv(file, ...)
}
env$View <- function(...) invisible(NULL)
lines <- readLines(source_file, warn=FALSE, encoding='UTF-8')
in_chunk <- FALSE; chunks <- list(); current <- character()
for (line in lines) {
  if (!in_chunk && grepl('^```\\{r', line)) {in_chunk <- TRUE; current <- character(); next}
  if (in_chunk && grepl('^```\\s*$', line)) {chunks[[length(chunks)+1L]] <- current; in_chunk <- FALSE; next}
  if (in_chunk) current <- c(current, line)
}
stopifnot(!in_chunk)
log_connection <- file(file.path(out, 'results', 'execution.log'), open='wt')
sink(log_connection, split=TRUE)
for (i in seq_along(chunks)) {
  cat('\n--- Archived R chunk', i, '---\n')
  eval(parse(text=chunks[[i]]), envir=env)
}
# The original notebook did not export RF 2024 predictions; export its existing vectors.
utils::write.csv(data.frame(pipe_id=env$test_2024$pipe_id, leak_date=env$test_2024$date,
 actual_cost=env$y_test_2024, predicted_raw=env$pred_raw_2024, predicted_log=env$pred_log_2024),
 'rf_predictions_2024.csv', row.names=FALSE)
sink(); close(log_connection)
# Combine the saved predictions and recompute every metric independently.
all_predictions <- list()
add <- function(file, model, variant, column) {
  d <- utils::read.csv(file, stringsAsFactors=FALSE)
  all_predictions[[length(all_predictions)+1L]] <<- data.frame(model=model, variant=variant,
    test_year=as.integer(substr(d$leak_date,1,4)), pipe_id=d$pipe_id,
    leak_date=d$leak_date, actual_cost=d$actual_cost, predicted_cost=d[[column]])
}
for (year in 2023:2026) {
  add(paste0('rf_predictions_',year,'.csv'),'Random Forest','raw','predicted_raw')
  if(year<2026) add(paste0('rf_predictions_',year,'.csv'),'Random Forest','log1p','predicted_log')
  add(paste0('gamma_train_2019_',year-1,'_test_',year,'_predictions.csv'),'Gamma','log link','predicted_gamma')
  if(year<2026) {
    f <- paste0('lightgbm_train_2019_',year-1,'_test_',year,'_predictions.csv')
    add(f,'LightGBM','raw','predicted_raw'); add(f,'LightGBM','log1p','predicted_log')
  } else add('lightgbm_log_train_2019_2025_test_2026_predictions.csv','LightGBM','log1p','predicted_log')
}
pred <- do.call(rbind, all_predictions)
stopifnot(all(is.finite(pred$predicted_cost)), !anyDuplicated(pred[c('model','variant','pipe_id')]))
data <- utils::read.csv(input)
idx <- match(pred$pipe_id, data$pipe_id)
stopifnot(!anyNA(idx), all(pred$actual_cost == data$cost[idx]))
groups <- split(pred, interaction(pred$model,pred$variant,pred$test_year,drop=TRUE))
metrics <- do.call(rbind,lapply(groups,function(d) data.frame(model=d$model[1],variant=d$variant[1],
 training_years=paste0('2019-',d$test_year[1]-1),test_year=d$test_year[1],n_test=nrow(d),
 MAE=mean(abs(d$actual_cost-d$predicted_cost)),RMSE=sqrt(mean((d$actual_cost-d$predicted_cost)^2)),
 total_bias_pct=100*(sum(d$predicted_cost)/sum(d$actual_cost)-1))))
metrics <- metrics[order(metrics$model,metrics$variant,metrics$test_year),]; rownames(metrics)<-NULL
averages <- do.call(rbind,lapply(split(metrics,interaction(metrics$model,metrics$variant,drop=TRUE)),function(d)
 data.frame(model=d$model[1],variant=d$variant[1],years=paste(d$test_year,collapse=','),n_years=nrow(d),
 mean_MAE=mean(d$MAE),mean_RMSE=mean(d$RMSE))))
utils::write.csv(metrics,file.path(out,'results','fold_metrics.csv'),row.names=FALSE)
utils::write.csv(averages,file.path(out,'results','average_metrics.csv'),row.names=FALSE)
utils::write.csv(pred,file.path(out,'results','all_predictions.csv'),row.names=FALSE)
for (spec in list(c('Random Forest','raw','output_randomforest.csv'),c('Gamma','log link','output_gamma.csv'),
                 c('LightGBM','log1p','output_lightgbm.csv'))) {
  d <- pred[pred$model==spec[1] & pred$variant==spec[2],c('pipe_id','predicted_cost')]
  stopifnot(nrow(d)==8586L,!anyDuplicated(d$pipe_id))
  utils::write.csv(d,file.path(out,spec[3]),row.names=FALSE)
}
jsonlite::write_json(list(input_sha256=expected,source_sha256=digest::digest(file=source_file,algo='sha256'),
 original_chunks_executed=length(chunks),prediction_rows=nrow(pred),metric_rows=nrow(metrics),
 primary_output_rows=8586,duplicate_model_variant_pipe_ids=0,actual_costs_match_source=TRUE),
 file.path(out,'results','verification.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'results','sessionInfo.txt'))
jsonlite::write_json(list(R=R.version.string,ranger=as.character(packageVersion('ranger')),
 lightgbm=as.character(packageVersion('lightgbm'))),file.path(out,'results','versions.json'),pretty=TRUE,auto_unbox=TRUE)
setwd(old)
cat('\nComplete:',out,'\n'); print(metrics)
