data {
    int num_nodes, num_dims;
    array [(num_nodes - 1) * num_nodes %/% 2] int adjacency;
    real bias_loc;
    real<lower=0> bias_scale, scale_conc, scale_rate;
}

transformed data {
    int num_pairs = (num_nodes - 1) * num_nodes %/% 2;
}

parameters {
    real bias;
    real<lower=0> scale;
    array [num_nodes] vector [num_dims] x;
}

model {
    array [num_pairs] real logits;
    int z = 1;
    for (i in 1:num_nodes) {
        for (j in i + 1: num_nodes) {
            logits[z] = bias - distance(x[i], x[j]);
            z += 1;
        }
    }

    for (i in 1:num_nodes) {
        x[i] ~ normal(0, scale);
    }
    bias ~ normal(bias_loc, bias_scale);
    scale ~ gamma(scale_conc, scale_rate);
    adjacency ~ bernoulli_logit(logits);
}
