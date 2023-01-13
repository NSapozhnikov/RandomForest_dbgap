## Helpers for PCA
suppressPackageStartupMessages(require(dplyr))
suppressPackageStartupMessages(require(ggplot2))

PlotGroupSizes <- function(data, output = NULL) {
  gs <- table(data$groups) %>% as.data.frame()
  ggplot(gs) + geom_bar(aes(x = Var1, y = Freq), stat="identity") +
    labs(x = "", y = "Size") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size = 10),
          axis.text.y = element_text(size = 10))
  if (!is.null(output)) {
    ggsave(output, device = "png", width = 8, height = 6, bg = "white")
  }
}

PlotEigenvalues <- function(data, output = NULL) {
  eval <- data[["eval"]]
  eval$n <- stringr::str_extract(eval$pc, "([0-9]+)")
  eval$n <- as.integer(eval$n)
  
  ggplot(eval) + geom_bar(aes(x = n, y = p), stat="identity") +
    labs(x = "PC", y = "Contribution, %") +
    theme_bw() +
    theme(axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10))
  
  if(is.null(output)) { return(p) } else { 
    ggsave(output, device = "png", width = 8, height = 6, bg = "white") }
}

PlotPCA <- function(data, tlt = NULL, output = NULL) {
  
  evec <- data[["evec"]]
  groups <- data[["groups"]]
  eval <- data[["eval"]]
 
  set.seed(1)
  group.labels <- unique(groups)
  n <- length(group.labels)
  palette <- randomcoloR::distinctColorPalette(n) 
  
  # Plot Principal Components
  pc <- list(c("PC1", "PC2"), c("PC1", "PC3"), c("PC3", "PC2"))
  
  p <- lapply(pc, function(x) {
    ggplot(evec) + 
      geom_point(aes(x = evec[, x[1]], 
                     y = evec[, x[2]], col = groups)) + 
      labs(x = sprintf("%s (%s%%)", x[1], eval$p[which(eval$pc == x[1])]),  
           y = sprintf("%s (%s%%)", x[2], eval$p[which(eval$pc == x[2])])) +
      scale_color_manual(values = palette) +
      theme(legend.position = "none", 
            panel.background = element_rect(fill = "white", color= "grey"),
            panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank())
  })
  
 fg <- ggpubr::ggarrange(plotlist = p, ncol = 2, nrow = 2, 
                    common.legend = TRUE, legend = "bottom")
 
 if(!is.null(tlt)) ggpubr::annotate_figure(fg, top = tlt)
 
 if(is.null(output)) { return(fg) } else { 
     ggsave(output, device = "png", width = 8, height = 8, bg = "white") }
}

LoadData1 <- function(fam, eigenvec){
  # Args:
  #  fam: path/to/filename.fam 
  #  eigenvec: path/to/filename.eigenvec file 
  
  # Initiate output
  out <- vector("list")
  
  # Load PCA results
  evec <- data.frame(read.table(eigenvec, header = FALSE, skip = 0, sep = " "))
  rownames(evec) <- evec[, 2]
  evec <- evec[, 3:ncol(evec)]
  colnames(evec) <- paste("PC", c(1:20), sep="")
  
  # Load fam file of data used for PCA
  status <- read.table(fam)[, c("V6", "V2")]
  
  # Compile the output
  out[["evec"]] <- evec
  t <- sapply(rownames(evec), function(x) status$V6[which(status$V2 == x)])
  out[["groups"]] <- as.factor(t)
  
  out
  
}

LoadData2 <- function(pref){
  # Args:
  #  pref: path/to/pref of bed/bim/fam Plink files
  
  # Initiate 
  out <- vector("list")
  fam.file <- sprintf("%s.fam", pref)
  eigenvec <- sprintf("%s.eigenvec", pref)
  eigenval <- sprintf("%s.eigenval", pref)
  
  # Check input
  if(!file.exists(fam.file)) stop("File ", fam.file, " doesn't exist!", call. = F)
  if(!file.exists(eigenvec)) stop("File ", eigenvec, " doesn't exist!", call. = F)
  if(!file.exists(eigenval)) stop("File ", eigenval, " doesn't exist!", call. = F)
  
  # Load PCA results
  evec <- data.frame(read.table(eigenvec, header = FALSE, skip = 0, sep = " "))
  rownames(evec) <- evec[, 2]
  evec <- evec[, 3:ncol(evec)]
  colnames(evec) <- paste("PC", c(1:20), sep="")
  
  eval <- data.frame(read.table(eigenval, header = FALSE, skip = 0, sep = " "))
  colnames(eval) <- c("v")
  eval$p <- round(eval$v/(sum(eval$v))*100)
  eval$cs <- cumsum(eval$p)
  eval$pc <- paste0("PC", 1:nrow(eval))
  
  # Load fam file of data used for PCA
  fam <- read.table(fam.file)
  
  # Compile the output
  out[["evec"]] <- evec
  out[["eval"]] <- eval
  out[["groups"]] <- as.factor(fam$V6)
  
  out
}


FindOutliers <- function(data) {
  # Subset the eigenvectors
  eigenvec <- data[["evec"]]
  
  # Find out the indexes of the outliers applying the rule 
  # "more than 6 standard deviations away from the mean"
  ind <- apply(eigenvec, 2, function(x) {
    which(abs(x - mean(x)) > 6 * sd(x))
  }) %>% Reduce(union, .)
  
  # Show outlier ids
  rownames(eigenvec)[ind]
}





