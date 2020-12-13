# Title     : Subset
# Objective : Visualize the relationship between DNA methylation and RNA expression
#             using a scatterplot
# Created by: kevin
# Created on: 12/9/2020

library(readr)
library(methylationfun)
library(reshape2)
library(dplyr)


# read in meth and rna icgc_donor_id df
temp <- temp[match(id$icgc_donor_id, temp$icgc_donor_id),]

# get row indexes of matching donors
row_idx <- which(meth_id$icgc_donor_id %in% rna_id$icgc_donor_id)

df <- as.data.frame(df)
col_idx <- which(grepl('_',df[1,]))


for(idx in col_idx){
  curr <- df[1,idx]
  df[1,idx] <- str_split(curr,'_',simplify = TRUE)[1]
}


df[1,row_idx] <- NaN

# subset methylation data
meth <- meth[row_idx,]
meth_id <- meth_id[row_idx,]


df <- as.data.frame(df)
num <- sapply(df[1,],is.numeric)
non.numeric.cols <- which(!num)
for(col in non.numeric.cols){
  for(row in seq(nrow(df))){
    if(!is.numeric(df[row,col])){
      df[row,col] <- 0
    }
  }
}

sum <- 0
for(i in seq(ncol(df_new))){
  sum <- sum + is.numeric(df[,i])
}



