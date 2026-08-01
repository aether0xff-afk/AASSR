# Causal Prophecy v2 targets and uncertainty

Protocol v2.0 predicts only observable next state, visible transition deltas,
action unlock, and discounted terminal-return probability.  The return head is
trained after episode termination from the observed binary outcome; private
viability is neither an input nor a target.

The default Monte Carlo target at step `t` is
`gamma^(T-t) * terminal_success`.  TD return targets are an explicit config
alternative and may not be pooled with Monte Carlo results.

Protocol v2.1 adds resource cost, observed damage, uncertainty, OOD score, and
calibration confidence.  The empirical implementation defines:

- count uncertainty: `1 / sqrt(visits + 1)`;
- holdout proxy: rolling observable-effect prediction error;
- uncertainty: equal-weight mean of count uncertainty and holdout error;
- OOD score: `1 / (visits + 1)`;
- calibration confidence: `1 - mean(uncertainty, rolling return Brier error)`.

All values are clipped to `[0, 1]`.  Reliability bins default to 10 and are
part of the resolved config.  Neural models must implement the same outputs
and are not enabled until the empirical v2.0 diagnostic has been recorded.
