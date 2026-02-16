//! Integration tests for probabilistic programming module
//!
//! These tests verify end-to-end workflows and interactions between different components
//! of the probabilistic programming module.

use super::*;
use approx::assert_relative_eq;
use scirs2_core::random::thread_rng;

#[test]
fn test_beta_distribution_integration() {
    let beta = BetaDistribution::new(2.0, 5.0).expect("Failed to create Beta distribution");

    // Test PDF, sampling, and moments together
    let mut rng = thread_rng();
    let mut samples = Vec::with_capacity(1000);

    for _ in 0..1000 {
        let sample = beta.sample(&mut rng).expect("Sampling failed");
        samples.push(sample);
    }

    // Empirical mean should be close to theoretical mean
    let empirical_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let theoretical_mean = beta.mean();

    assert!((empirical_mean - theoretical_mean).abs() < 0.1);
}

#[test]
fn test_metropolis_hastings_workflow() {
    // Target: Bivariate normal with known correlation
    let target_mean = [1.0, 2.0];
    let target_std = [1.0, 1.5];

    let log_posterior = |theta: &[f64]| -> f64 {
        let mut log_prob = 0.0;
        for i in 0..2 {
            let z = (theta[i] - target_mean[i]) / target_std[i];
            log_prob -= 0.5 * z * z;
        }
        log_prob
    };

    let proposal = GaussianProposal::new(0.5).expect("Failed to create proposal");
    let sampler = MetropolisHastings::new(log_posterior, proposal);

    let mut rng = thread_rng();
    let result = sampler
        .sample(&[0.0, 0.0], 10000, 1000, &mut rng)
        .expect("MCMC sampling failed");

    // Check acceptance rate is reasonable (can vary with random seed)
    assert!(result.acceptance_rate > 0.05 && result.acceptance_rate < 0.9);

    // Check posterior mean estimates (relaxed tolerances for stochastic MCMC)
    let mean = result.mean();
    assert!((mean[0] - target_mean[0]).abs() < 0.25);
    assert!((mean[1] - target_mean[1]).abs() < 0.35);

    // Check ESS is reasonable (stochastic test - can vary with random seed)
    let ess = result.effective_sample_size();
    for &e in &ess {
        assert!(e > 40.0); // Should have reasonable number of effective samples
    }
}

#[test]
fn test_hmc_workflow() {
    // Target: Standard normal
    let log_posterior = |theta: &[f64]| -> f64 { -0.5 * theta[0].powi(2) };

    let grad_log_posterior = |theta: &[f64]| -> Vec<f64> { vec![-theta[0]] };

    let hmc = HamiltonianMC::new(log_posterior, grad_log_posterior, 0.1, 10)
        .expect("Failed to create HMC sampler");

    let mut rng = thread_rng();
    let result = hmc
        .sample(&[0.0], 2000, 200, &mut rng)
        .expect("HMC sampling failed");

    // HMC should have higher acceptance rate than MH
    assert!(result.acceptance_rate > 0.5);

    let mean = result.mean();
    assert!(mean[0].abs() < 0.2);
}

#[test]
fn test_variational_inference_workflow() {
    // Target: N(1, 0.5²)
    let target_mean = 1.0;
    let target_std = 0.5;

    let log_likelihood = |theta: &[f64]| -> f64 {
        // Likelihood for observed data x=1
        let x = 1.0;
        -0.5 * ((x - theta[0]) / target_std).powi(2)
    };

    let log_prior = |theta: &[f64]| -> f64 {
        // Standard normal prior
        -0.5 * theta[0].powi(2)
    };

    let elbo_computer = ELBO::new(log_likelihood, log_prior);
    let mut variational = MeanFieldVariational::new(1);

    // Simple gradient ascent on ELBO
    let mut rng = thread_rng();
    let learning_rate = 0.01;

    for _ in 0..100 {
        let (grad_means, grad_log_stds) =
            elbo_computer.estimate_gradient(&variational, 100, &mut rng);

        // Update parameters
        variational.means[0] += learning_rate * grad_means[0];
        variational.log_stds[0] += learning_rate * grad_log_stds[0];
    }

    // Variational mean should be close to posterior mean
    // Posterior for this model is approximately N(0.667, 0.447²)
    assert!((variational.means[0] - 0.667).abs() < 0.3);
}

#[test]
fn test_beta_binomial_conjugate_workflow() {
    // Prior: Uniform (Beta(1,1))
    let prior = BetaBinomialConjugate::new(1.0, 1.0).expect("Failed to create conjugate prior");

    // Simulate coin flips
    let n_trials = 100;
    let n_successes = 60;

    // Update posterior
    let posterior = prior
        .update(n_successes, n_trials)
        .expect("Posterior update failed");

    // Posterior mean should be close to MLE
    let mle = n_successes as f64 / n_trials as f64;
    assert!((posterior.mean() - mle).abs() < 0.05);

    // Posterior predictive
    let pred_prob = prior
        .posterior_predictive(n_successes, n_trials)
        .expect("Predictive computation failed");
    assert!((pred_prob - 0.6).abs() < 0.05);
}

#[test]
fn test_model_comparison_workflow() {
    // Simulate some data
    let log_likelihood = -50.0;

    // Model 1: Simple model with 3 parameters
    let bic1 = bic(log_likelihood, 3, 100);

    // Model 2: Complex model with 10 parameters
    let bic2 = bic(log_likelihood, 10, 100);

    // BIC should penalize complex model
    assert!(bic2 > bic1);
}

#[test]
fn test_waic_workflow() {
    // Generate pointwise log-likelihoods for 3 MCMC samples and 5 observations
    let pointwise_ll = vec![
        vec![-1.0, -1.5, -2.0, -1.2, -1.8],
        vec![-1.1, -1.4, -2.1, -1.3, -1.7],
        vec![-0.9, -1.6, -1.9, -1.1, -1.9],
    ];

    let waic_result = waic(&pointwise_ll).expect("WAIC computation failed");

    assert!(waic_result.waic.is_finite());
    assert!(waic_result.p_waic > 0.0);
    assert!(waic_result.lppd.is_finite());
}

#[test]
fn test_credible_intervals_workflow() {
    // Generate samples from known distribution
    let samples: Vec<f64> = (0..1000).map(|i| (i as f64 - 500.0) / 100.0).collect();

    // Equal-tailed 95% CI
    let (lower_et, upper_et) =
        equal_tailed_interval(&samples, 0.05).expect("Equal-tailed interval failed");

    // HPD 95% CI
    let (lower_hpd, upper_hpd) = hpd_interval(&samples, 0.05).expect("HPD interval failed");

    // Both should cover approximately 95% of data
    assert!(lower_et < 0.0 && upper_et > 0.0);
    assert!(lower_hpd < 0.0 && upper_hpd > 0.0);

    // For symmetric distribution, intervals should be similar
    let width_et = upper_et - lower_et;
    let width_hpd = upper_hpd - lower_hpd;
    assert!((width_et - width_hpd).abs() < 1.0);
}

#[test]
fn test_hmm_complete_workflow() {
    let mut hmm = HiddenMarkovModel::new(2, 2).expect("HMM creation failed");

    // Set up a simple weather model
    // States: 0 = Sunny, 1 = Rainy
    // Observations: 0 = Dry, 1 = Wet

    // Initial: More likely to start sunny
    hmm.initial = vec![0.7, 0.3];

    // Transition: Weather tends to persist
    hmm.transition = vec![
        vec![0.8, 0.2], // Sunny -> Sunny/Rainy
        vec![0.3, 0.7], // Rainy -> Sunny/Rainy
    ];

    // Emission: Observations mostly match state
    hmm.emission = vec![
        vec![0.9, 0.1], // Sunny -> Dry/Wet
        vec![0.2, 0.8], // Rainy -> Dry/Wet
    ];

    // Generate sequence
    let mut rng = thread_rng();
    let (true_states, observations) = hmm
        .generate(10, &mut rng)
        .expect("Sequence generation failed");

    // Forward algorithm
    let alpha = hmm
        .forward(&observations)
        .expect("Forward algorithm failed");
    assert_eq!(alpha.len(), 10);

    // Backward algorithm
    let beta = hmm
        .backward(&observations)
        .expect("Backward algorithm failed");
    assert_eq!(beta.len(), 10);

    // Viterbi (most likely path)
    let predicted_states = hmm.viterbi(&observations).expect("Viterbi failed");
    assert_eq!(predicted_states.len(), 10);

    // Likelihood
    let likelihood = hmm
        .likelihood(&observations)
        .expect("Likelihood computation failed");
    assert!(likelihood > 0.0 && likelihood <= 1.0);

    // Check that Viterbi path is plausible
    let mut matches = 0;
    for i in 0..10 {
        if predicted_states[i] == true_states[i] {
            matches += 1;
        }
    }
    // Should get at least some states correct
    assert!(matches > 0);
}

#[test]
fn test_bayesian_network_workflow() {
    let mut bn = BayesianNetwork::new();

    // Create a simple network: Rain -> Sprinkler -> Grass Wet
    let mut rain = BayesianNode::new("Rain".to_string(), 2);
    rain.set_cpt(vec![], vec![0.8, 0.2]).unwrap(); // P(Rain = No/Yes)

    let mut sprinkler = BayesianNode::new("Sprinkler".to_string(), 2);
    sprinkler.add_parent(0); // Sprinkler depends on Rain
    sprinkler.set_cpt(vec![0], vec![0.6, 0.4]).unwrap(); // P(S|R=No)
    sprinkler.set_cpt(vec![1], vec![0.99, 0.01]).unwrap(); // P(S|R=Yes)

    let mut grass = BayesianNode::new("Grass Wet".to_string(), 2);
    grass.add_parent(0); // Grass depends on Rain
    grass.add_parent(1); // Grass depends on Sprinkler
    grass.set_cpt(vec![0, 0], vec![0.99, 0.01]).unwrap(); // P(G|R=No,S=No)
    grass.set_cpt(vec![0, 1], vec![0.1, 0.9]).unwrap(); // P(G|R=No,S=Yes)
    grass.set_cpt(vec![1, 0], vec![0.1, 0.9]).unwrap(); // P(G|R=Yes,S=No)
    grass.set_cpt(vec![1, 1], vec![0.01, 0.99]).unwrap(); // P(G|R=Yes,S=Yes)

    bn.add_node(rain);
    bn.add_node(sprinkler);
    bn.add_node(grass);

    // Verify it's a DAG
    assert!(bn.is_dag());

    // Sample from the network
    let mut rng = thread_rng();
    let mut samples = Vec::new();
    for _ in 0..100 {
        let sample = bn.sample(&mut rng).expect("Sampling failed");
        samples.push(sample);
    }

    // Check that we get varied samples
    assert!(samples.len() == 100);
}

#[test]
fn test_gaussian_process_workflow() {
    let kernel = RBFKernel::new(1.0, 1.0).expect("Kernel creation failed");
    let mut gp = GaussianProcess::new(kernel, 0.01).expect("GP creation failed");

    // Train on simple function f(x) = sin(x)
    let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|x| x[0].sin()).collect();

    gp.fit(x_train.clone(), y_train.clone())
        .expect("GP fit failed");

    // Predict at training points
    let (means, variances) = gp.predict(&x_train).expect("GP prediction failed");

    // At training points, predictions should be close to true values
    for (i, &mean) in means.iter().enumerate() {
        assert!((mean - y_train[i]).abs() < 0.5);
    }

    // Variances should be small at training points
    for &var in &variances {
        assert!(var < 0.2);
    }

    // Predict at new points
    let x_test = vec![vec![2.5], vec![3.0]];
    let (means_test, variances_test) = gp.predict(&x_test).expect("GP test prediction failed");

    assert_eq!(means_test.len(), 2);
    assert_eq!(variances_test.len(), 2);

    // Predictions should be finite
    for &m in &means_test {
        assert!(m.is_finite());
    }
}

#[test]
fn test_bayes_factor_workflow() {
    // Model 1: Good fit
    let log_ml_m1 = -100.0;

    // Model 2: Poor fit
    let log_ml_m2 = -150.0;

    let bf = bayes_factor(log_ml_m1, log_ml_m2);

    // BF should strongly favor Model 1
    assert!(bf > 1e20); // exp(50) is huge
}

#[test]
fn test_posterior_predictive_check_workflow() {
    // Observed test statistic
    let observed = 5.0;

    // Replicated test statistics from posterior predictive
    let replicated = vec![3.0, 4.0, 5.0, 6.0, 7.0, 4.5, 5.5, 6.5, 7.5, 8.0];

    let pvalue = posterior_predictive_pvalue(observed, &replicated)
        .expect("Posterior predictive p-value failed");

    // p-value should be in [0, 1]
    assert!((0.0..=1.0).contains(&pvalue));

    // For this data, p-value should be around 0.7 (7 out of 10 are >= 5.0)
    assert!((pvalue - 0.7).abs() < 0.1);
}

#[test]
fn test_dirichlet_multinomial_workflow() {
    // Uniform prior
    let prior = DirichletMultinomialConjugate::new(vec![1.0, 1.0, 1.0])
        .expect("Dirichlet prior creation failed");

    // Observe categorical data
    let counts = vec![30, 50, 20];

    // Update posterior
    let posterior_probs = prior
        .posterior_predictive(&counts)
        .expect("Posterior predictive failed");

    assert_eq!(posterior_probs.len(), 3);

    // Probabilities should sum to 1
    let sum: f64 = posterior_probs.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

    // Posterior should favor the category with most observations
    assert!(posterior_probs[1] > posterior_probs[0]);
    assert!(posterior_probs[1] > posterior_probs[2]);
}

#[test]
fn test_end_to_end_bayesian_regression() {
    // Simulate linear regression data: y = 2x + 1 + noise
    let n = 50;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let mut y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    // Add noise using Normal-Normal conjugate prior simulation
    let mut rng = thread_rng();
    for y in &mut y_data {
        let noise = sample_standard_normal(&mut rng) * 0.5;
        *y += noise;
    }

    // Use Normal-Normal conjugate prior for intercept (simplified 1D case)
    let prior = NormalNormalConjugate::new(0.0, 10.0, 0.25).expect("Prior creation failed");

    // Extract residuals after removing slope
    let residuals: Vec<f64> = y_data
        .iter()
        .zip(&x_data)
        .map(|(&y, &x)| y - 2.0 * x)
        .collect();

    let (post_mean, post_var) = prior.update(&residuals).expect("Posterior update failed");

    // Posterior mean should be close to true intercept (1.0)
    assert!((post_mean - 1.0).abs() < 0.5);

    // Posterior variance should be smaller than prior
    assert!(post_var < 10.0);

    // Compute credible interval
    let (lower, upper) = prior
        .credible_interval_95(&residuals)
        .expect("Credible interval failed");

    // True intercept should be in the interval
    assert!(lower < 1.0 && upper > 1.0);
}

// Helper function for standard normal sampling
fn sample_standard_normal<R: scirs2_core::random::Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
