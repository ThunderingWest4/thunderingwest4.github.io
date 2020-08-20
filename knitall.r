files <- c("home.rmd", "huntthewumpus.rmd", "irisflower.rmd", "underconstruction.rmd")
for (f in files) { rmarkdown::render(f) }