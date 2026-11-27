// erf to within 1.5e-7 (Abramowitz and Stegun 7.1.26).
float erf_approx(float x) {
    const float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    const float poly =
        t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-x * x));
}

float gelu(float x) {
    return 0.5 * x * (1.0 + erf_approx(x * 0.70710678118654752));
}
