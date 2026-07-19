import numpy as np

from audit_da.bayes import ApproxHierarchicalBayes


def test_hierarchical_bayes_produces_finite_posterior():
    rng = np.random.default_rng(7)
    firms = np.repeat(["A", "B", "C", "D"], 40)
    x = rng.normal(size=(len(firms), 2))
    firm_effect = {"A": 0.1, "B": -0.1, "C": 0.05, "D": -0.05}
    y = 0.2 + x @ np.array([0.4, -0.3]) + np.array([firm_effect[f] for f in firms]) + rng.normal(0, 0.05, len(firms))
    model = ApproxHierarchicalBayes().fit(x, y, firms, ["x1", "x2"])
    prediction = model.posterior_mean_sd(x[:10], firms[:10])
    assert np.isfinite(prediction.mean).all()
    assert (prediction.sd > 0).all()
    coef, firm_draws = model.draw_components(100, rng)
    latent = model.latent_draws(x[:10], firms[:10], coef, firm_draws)
    assert latent.shape == (10, 100)
