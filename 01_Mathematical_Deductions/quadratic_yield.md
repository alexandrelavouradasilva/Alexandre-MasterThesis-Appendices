# Deduction of Equation: Quadratic Equivalent Yield Stress

Recalling the macroscopic yield condition previously introduced:

$$ B \left[ \left(\left|\Sigma_{1}\right|-k\Sigma_{1}\right)^a + \left(\left|\Sigma_{2}\right|-k\Sigma_{2}\right)^a + \left(\left|\Sigma_{3}\right|-k\Sigma_{3}\right)^a \right]^{\frac{1}{a}} = Y $$

For the specific case where the homogeneity exponent is $a=2$ and the strength differential parameter is disabled ($k=0$), the equivalent yield stress simplifies significantly to:

$$ Y = B \sqrt{\Sigma_1^2 + \Sigma_2^2 + \Sigma_3^2} $$

Squaring both sides of the equation yields:

$$ Y^2 = B^2 (\Sigma_1^2 + \Sigma_2^2 + \Sigma_3^2) $$

The sum of the squared transformed stresses can be represented in vector notation as the dot product $\Sigma^T \Sigma$. Given that the transformed stress vector $\Sigma$ is defined by the linear transformation of the Cauchy stress tensor $\sigma$ via the normal block of the constitutive matrix $C_n$ (i.e., $\Sigma = C_n \sigma$), the quadratic term expands to:

$$ \Sigma^T \Sigma = (C_n \sigma)^T (C_n \sigma) = \sigma^T C_n^T C_n \sigma $$

Substituting this expansion back into the yield function reveals the underlying quadratic form:

$$ Y^2 = B^2 \left( \sigma^T C_n^T C_n \sigma \right) = \sigma^T \underbrace{\left(B^2 C_n^T C_n\right)}_{H} \sigma $$
