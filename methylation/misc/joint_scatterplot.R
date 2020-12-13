# Title     : Joint Scatterplot
# Objective : Visualize the relationship between DNA methylation and RNA expression
#             using a scatterplot
# Created by: kevin
# Created on: 12/9/2020

library(readr)
library(BBmisc)
library(ggplot2)

# get rna data
rna <- read_csv('exp_seq.csv')
rna.id <- read_csv('icgc_donor_id.csv')
rna.genes <- read_csv('gene_id.csv')
colnames(rna) <- rna.genes$gene_id

# get methylation data
meth <- read_csv('meth_array.csv')
meth.id <- read_csv('icgc_donor_id.csv')
meth.genes <- data.frame(gene_id = colnames(meth))


# subset so both rna and methylation contain exactly the same genes and patients
match <- intersect(rna.genes$gene_id,meth.genes$gene_id)

names.use <- which(rna.genes$gene_id %in% match)
rna.subset <- rna[,names.use]

names.use <- which(meth.genes$gene_id %in% match)
meth.subset <- meth[, names.use]

rna.subset <- as.data.frame(rna.subset)
rownames(rna.subset) <- rna.id$icgc_donor_id

meth.subset <- as.data.frame(meth.subset)
rownames(meth.subset) <- meth.id$icgc_donor_id

rows.use.index <- which(rownames(meth.subset) %in% rownames(rna.subset))
rows.use <- rownames(meth.subset)[rows.use.index]

meth.final <- meth.subset[rownames(meth.subset) %in% rows.use,]
rna.final <- normalize(rna.subset[rownames(rna.subset) %in% rows.use,])

meth.vec <- as.vector(t(meth.final))
rna.vec <- as.vector(t(rna.final))

meth.rna.df <- data.frame(methylation_value = meth.vec,
                          expression_value = rna.vec)


# plot
plotted<-ggplot(meth.rna.df,aes(x=methylation_value,y=expression_value))+
  xlab('methylation value') +
  ylab('RNA expression value') +
  ggtitle('DNA methylation + RNA expression') +
  geom_point()
plotted


#df <- meth.final
#df2 <- rna.final
#
#
#df$row.names<-rownames(df)
#long.df<-melt(df,id=c("row.names"))
#df2$row.names<-rownames(df2)
#long.df2<-melt(df2,id=c("row.names"))
#
#plotted<-ggplot(long.df,aes(x=variable,y=row.names,color=value))+
#  xlab('genes') +
#  ylab('donors') +
#  ggtitle('Methylation') +
#  geom_point() +
#  theme(
#        axis.text.x=element_blank(),
#        axis.ticks.x=element_blank()) +
#  theme(
#        axis.text.y=element_blank(),
#        axis.ticks.y=element_blank())
#plotted
#
#
#plotted2<-ggplot(long.df2,aes(x=variable,y=row.names,color=value))+
#  xlab('genes') +
#  ylab('donors') +
#  ggtitle('RNA') +
#  geom_point() +
#  theme(
#    axis.text.x=element_blank(),
#    axis.ticks.x=element_blank()) +
#  theme(
#    axis.text.y=element_blank(),
#    axis.ticks.y=element_blank())
#plotted2