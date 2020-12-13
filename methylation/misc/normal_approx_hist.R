library(tidyr)
library(ggplot2)
library(readr)
library(methylationfun)
library(nortest)

# read in data
n <- 200000  # specify how many data points to sample
methylation_values <- read_csv('methylation_values.csv')$methylation_value
values <- sample(methylation_values,n,replace=FALSE)

alfers <- alfers_dinges_approx(values)
wise <- wise_approx(values)
peizer <- peizer_pratt_approx(value)



# original 
bw <- 2 * IQR(values) / length(values)^(1/3)
ggplot(data.frame(methylation_value=values), aes(x=methylation_value)) + 
  geom_histogram(color="black", fill="lightblue", binwidth = bw) +
  xlab('methylation value') +
  ylab('frequency') +
  labs(title = 'Raw Data') +
  theme(plot.title = element_text(hjust = 0.5))


# alfers dinges 
bw <- 2 * IQR(alfers) / length(alfers)^(1/3)
ggplot(data.frame(methylation_value=alfers), aes(x=methylation_value)) + 
  geom_histogram(color="black", fill="lightblue", binwidth = bw) +
  xlab('methylation value') +
  ylab('frequency') +
  labs(title = 'Alfers Dinges Approximation') +
  theme(plot.title = element_text(hjust = 0.5))


#peizer
bw <- 2 * IQR(peizer) / length(peizer)^(1/3)
ggplot(data.frame(methylation_value=peizer), aes(x=methylation_value)) + 
  geom_histogram(color="black", fill="lightblue", binwidth = bw) +
  labs(title = "Peizer Platt Approximation") +
  xlab('methylation value') + 
  theme(plot.title = element_text(hjust = 0.5))


# Wise
bw <- 2 * IQR(wise) / length(wise)^(1/3)
ggplot(data.frame(methylation_value=wise), aes(x=methylation_value)) + 
  geom_histogram(color="black", fill="lightblue", binwidth = bw) +
  labs(title = "Wise Approximation") +
  xlab('methylation value') + 
  theme(plot.title = element_text(hjust = 0.5))


# qqplot
raw<-ggplot(data.frame(x=values), aes(sample=x)) +
  stat_qq(colour='lightblue') +
  labs(title = "Raw Data") +
  theme(plot.title = element_text(hjust = 0.5))
raw

alfers<-ggplot(data.frame(x=alfers), aes(sample=x)) +
  stat_qq(colour='lightblue') +
  labs(title = "Alfers Dinges Approximation") +
  theme(plot.title = element_text(hjust = 0.5))
alfers

wise<-ggplot(data.frame(x=wise), aes(sample=x)) +
  stat_qq(colour='lightblue') +
  labs(title = "Wise Approximation") +
  theme(plot.title = element_text(hjust = 0.5))
wise



# test the normal approximations
iter <- 500

# shapiro-wilk test (higher test statistic means more normal)
p.alfers.shapiro <- 0
p.wise.shapiro <- 0
w.alfers.shapiro <- 0
w.wise.shapiro <- 0
for(i in 1:iter){
  p.alfers.shapiro <- p.alfers.shapiro + shapiro.test(sample(alfers,5000,replace=FALSE))$p.value
  p.wise.shapiro <- p.wise.shapiro + shapiro.test(sample(wise,5000,replace=FALSE))$p.value
  w.alfers.shapiro <- w.alfers.shapiro + shapiro.test(sample(alfers,5000,replace=FALSE))$statistic
  w.wise.shapiro<- w.wise.shapiro + shapiro.test(sample(wise,5000,replace=FALSE))$statistic
}

# anderson-darling test (lower test statistic means more normal)
p.alfers.ad <- 0
p.wise.ad <- 0
w.alfers.ad <- 0
w.wise.ad <- 0
for(i in 1:iter){
  p.alfers.ad <- p.alfers.ad + ad.test(sample(alfers,5000,replace=FALSE))$p.value
  p.wise.ad <- p.wise.ad + ad.test(sample(wise,5000,replace=FALSE))$p.value
  w.alfers.ad <- w.alfers.ad + ad.test(sample(alfers,5000,replace=FALSE))$statistic
  w.wise.ad <- w.wise.ad + ad.test(sample(wise,5000,replace=FALSE))$statistic
}

print("======= Shapiro-Wilks ========")
cat("Alfers Dinges p-value: ", p.alfers.shapiro/iter)
cat("Wise p-value: ", p.wise.shapiro/iter)
cat("Alfers Dinges W statistic: ", w.alfers.shapiro/iter)
cat("Wise W statistic: ", w.wise.shapiro/iter)

print("======= Anderson-Darling ========")
cat("Alfers Dinges p-value: ", p.alfers.ad/iter)
cat("Wise p-value: ", p.wise.ad/iter)
cat("Alfers Dinges W statistic: ", w.alfers.ad/iter)
cat("Wise W statistic: ", w.wise.ad/iter)


