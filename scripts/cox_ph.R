# Exact survival::coxph / cox.zph diagnostic on an explicitly prepared endpoint risk set.
# Usage: Rscript scripts/cox_ph.R RISKSET.csv 'Surv(time_years,event) ~ exposure_indicator + chronological_age + factor(sex_code)' OUTPUT_PREFIX
args <- commandArgs(trailingOnly=TRUE)
if (length(args)<3) stop("Pass risk set CSV, formula, output prefix")
if (!requireNamespace("survival",quietly=TRUE)) stop("Install survival")
data <- read.csv(args[1])
model <- survival::coxph(as.formula(args[2]),data=data,ties="efron",x=TRUE, control=survival::coxph.control(iter.max=100))
ph <- survival::cox.zph(model, transform=if (length(args)>=4) args[4] else "rank")
write.csv(as.data.frame(ph$table),paste0(args[3],"_cox_zph.csv"))
writeLines(capture.output(sessionInfo()),paste0(args[3],"_R_sessionInfo.txt"))
