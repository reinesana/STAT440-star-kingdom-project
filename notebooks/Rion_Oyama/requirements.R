# Run once in the R console. Do not install packages during knitting.
install.packages(c("ranger", "lightgbm", "jsonlite", "digest"),
                 repos = "https://cloud.r-project.org")
# Tested versions are recorded in results/versions.json and sessionInfo.txt.
