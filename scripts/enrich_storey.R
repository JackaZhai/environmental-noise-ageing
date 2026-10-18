# Usage: Rscript scripts/enrich_storey.R RESULTS_DIR GENE_SETS_GMT
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop("Pass results directory and GMT path")
if (!requireNamespace("clusterProfiler", quietly=TRUE) || !requireNamespace("qvalue", quietly=TRUE)) stop("Install clusterProfiler and qvalue in your R environment")
base <- file.path(args[1], "bridge_enrichment")
candidates <- read.csv(file.path(base, "enrichment_candidates.csv"))
universe <- read.csv(file.path(base, "enrichment_universe.csv"))
gmt <- clusterProfiler::read.gmt(args[2]); names(gmt) <- c("term", "gene")
outputs <- list()
for (contrast in unique(universe$contrast)) {
  genes <- unique(candidates$gene[candidates$contrast == contrast])
  background <- unique(universe$gene[universe$contrast == contrast])
  if (!length(genes)) next
  fit <- clusterProfiler::enricher(genes, universe=background, TERM2GENE=gmt,
       pvalueCutoff=1, qvalueCutoff=1, pAdjustMethod="BH", minGSSize=3, maxGSSize=500)
  if (is.null(fit)) next
  # slot retains all eligible tested terms rather than only significant display rows.
  table <- fit@result; table$contrast <- contrast
  if (!"qvalue" %in% names(table)) table$qvalue <- NA_real_
  table$storey_q <- table$qvalue
  table$q_status <- ifelse(is.finite(table$storey_q), "estimated", "not_estimable")
  outputs[[contrast]] <- table
}
if (length(outputs)) {
  write.csv(do.call(rbind, outputs), file.path(base, "enrichment_storey.csv"), row.names=FALSE)
} else {
  write.csv(data.frame(contrast=character(), ID=character(), storey_q=numeric(), q_status=character()), file.path(base,"enrichment_storey.csv"),row.names=FALSE)
}
writeLines(capture.output(sessionInfo()), file.path(base, "R_sessionInfo.txt"))
