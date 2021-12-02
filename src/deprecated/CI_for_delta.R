library(binom)
setwd("~/GitHub/group-testing-with-household-correlation/src")

df <- read.csv(file='../results/PCR_tests/bounds_in_Theorem_2.csv')

df['Xbar'] <- df['numerator'] / df['niters']
df['Zbar'] <- df['denominator'] / df['niters']


q <- 0.9999
z_score <- qnorm((1+q)/2)
df['L_Z'] <- df['Zbar'] - z_score * sqrt(df['Zbar'] * (1 - df['Xbar']) / df['niters'])
df['U_Z'] <- df['Zbar'] + z_score * sqrt(df['Zbar'] * (1 - df['Xbar']) / df['niters'])


CI_exact_lb <- function(x, n) {
  return(binom.confint(x, n, methods=c('exact'))['lower'])
}
CI_exact_ub <- function(x, n) {
  return(binom.confint(x, n, methods=c('exact'))['upper'])
}

df['L_X'] <- unlist(mapply(CI_exact_lb, df$numerator, df$niters))
df['U_X'] <- unlist(mapply(CI_exact_ub, df$numerator, df$niters))
df['delta_hat'] <- df['Xbar'] / df['Zbar']
df['delta_lb'] <- df['L_X'] / df['U_Z']
df['delta_ub'] <- df['U_X'] / df['L_Z']

df = subset(df, select=c('pool.size', 'LoD', 'Xbar', 'Zbar', 'delta_hat', 'delta_lb', 'delta_ub'))
df = format(df, digits=3, nsmall=3)
write.csv(df, file='../results/PCR_tests/bounds_in_Theorem_2_with_CI.csv', row.names=TRUE)
