

//Hierarchical - Now with covariants
data {
    int<lower=0> K; // number of features
    int<lower=0> N; // number of data points
    //int<lower=0> N_pred; // number of datapoints to predict
    int<lower=0> J; // number of countries
    int<lower=0> M; // number of years

    int<lower=1,upper=J> C[N]; // Country indicator
    int<lower=1,upper=M> T[N]; // Year indicator
    //int<lower=1,upper=J> C_pred[N_pred]; //Country indicator for obs to predict

    int y[N]; // dummy label
    matrix[N, K] X; // features
    //matrix[N_pred, K] X_pred; // features for prediction
}
parameters {
    vector[K] beta; // coefficients for predictors

    vector[J] alpha_raw1; // Country individual intercept
    vector[M] alpha_raw2; // Year individual intercept

    real mu_alpha;

    real<lower=0> sigma_alpha1; // Hierarchical country deviation
    real<lower=0> sigma_alpha2; // Hierarchical year deviation

}
transformed parameters {
    vector[J] alpha1 = mu_alpha + sigma_alpha1 * alpha_raw1 ; // Country intercept
    vector[M] alpha2 = mu_alpha + sigma_alpha2 * alpha_raw2 ; // Year intercept
}
model {
    beta ~ normal(0, 1); //

    mu_alpha ~ normal(0, 10); //

    sigma_alpha1 ~ normal(0, 10);//
    alpha_raw1 ~ normal(0, 10); //

    sigma_alpha2 ~ normal(0, 10);//
    alpha_raw2 ~ normal(0, 10); //

    y ~ bernoulli_logit(X * beta + alpha1[C] + alpha2[T]); // likelihood
}
generated quantities {
    vector[N] y_tilde;
    //vector[N_pred] y_pred; // new, remove if fuck up
    vector[N] log_lik;

    for (n in 1:N)
        y_tilde[n] = bernoulli_logit_rng(X[n] * beta + alpha1[C[n]] + alpha2[T[n]]); //

    //for (n in 1:N_pred)
    //    y_pred[n] = bernoulli_logit_rng(X_pred[n] * beta + alpha1[C_pred[n]] + mean(alpha2));

    for (n in 1:N)
        log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n]* beta + alpha1[C[n]] + alpha2[T[n]]); //
}

