import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.function
def boxcox_z(y, mu, sigma, nu, epsilon = 1.0e-4):
    # Tolerance for considering nu to be actually equal to zero
    is_nu_zero = (tf.math.abs(nu) < epsilon)
    
    # If nu is zero, replace it by 1.0e-7 to prevent division by zero
    safe_nu = tf.where(tf.equal(nu, 0.0), tf.constant(epsilon, dtype = tf.float32), nu)
    
    # Box-Cox transformation
    pow_case = (1.0 / (sigma * safe_nu)) * (tf.pow(y / mu, safe_nu) - 1.0)
    log_case = (1.0 / sigma) * ( tf.math.log(y) - tf.math.log(mu) )
    z = tf.where(is_nu_zero, log_case, pow_case)

    # Prevent z from ever reaching 'inf' so w*z never evaluates to 0.0 * inf.
    z_safe = tf.clip_by_value(z, -1.0e4, 1.0e4)
    
    return z
    
@tf.custom_gradient
def log_pdf_dmu(y, mu, sigma, nu, tau):
    z = boxcox_z(y, mu, sigma, nu)
    w = (tau+1.0)/(tau+z**2.0)
    first_derivative_mu = (w*z)/(sigma*mu) + nu/mu*(w*z**2.0 - 1.0)

    def custom_second_derivative(upstream_grad):
        # We explicitly tells tensorflow the derivative of f(x) = x^3 is a constant function equal to 5        
        d2log_pdf_y_mu2 = -(tau + 2.0*sigma**2.0*nu**2.0*tau + 1.0) / (mu**2.0*sigma**2.0*(tau+3.0))
        d2log_pdf_y_musigma = -2.0*nu*tau / (mu*sigma*(tau+3.0))
        d2log_pdf_y_munu = (tau-3.0) / (2.0*mu*(tau+3.0))
        d2log_pdf_y_mutau = 2.0*nu / (mu*(tau+1.0)*(tau+3.0))

        jac_mu2 = upstream_grad * d2log_pdf_y_mu2
        jac_musigma = upstream_grad * d2log_pdf_y_musigma
        jac_munu = upstream_grad * d2log_pdf_y_munu
        jac_mutau = upstream_grad * d2log_pdf_y_mutau
        
        # Return the derivatives of log_f_y with respect to all arguments (y is None since it is not our interest to differentiate in y)
        return None, jac_mu2, jac_musigma, jac_munu, jac_mutau
    
    return first_derivative_mu, custom_second_derivative

@tf.custom_gradient
def log_pdf_dsigma(y, mu, sigma, nu, tau):
    z = boxcox_z(y, mu, sigma, nu)
    w = (tau+1.0)/(tau+z**2.0)
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)

    arg_t = 1.0 / (sigma * tf.math.abs(nu))
    h_sigma_nu_tau = dist.prob( arg_t ) / dist.cdf( arg_t )

    first_derivative_sigma = 1.0/sigma * (w*z**2.0 - 1.0) + h_sigma_nu_tau/(sigma**2.0*tf.math.abs(nu))

    def custom_second_derivative(upstream_grad):
        # We explicitly tells tensorflow the derivative of f(x) = x^3 is a constant function equal to 5        
        d2log_pdf_y_sigma2 = -2.0*tau / (sigma**2*(tau+3.0))
        d2log_pdf_y_musigma = -2.0*nu*tau / (mu*sigma*(tau+3.0))
        d2log_pdf_y_sigmanu = -sigma*nu*tau / (tau+3.0)
        d2log_pdf_y_sigmatau = 2.0 / (sigma*(tau+1.0)*(tau+3.0))

        jac_sigma2 = upstream_grad * d2log_pdf_y_sigma2
        jac_musigma = upstream_grad * d2log_pdf_y_musigma
        jac_sigmanu = upstream_grad * d2log_pdf_y_sigmanu
        jac_sigmatau = upstream_grad * d2log_pdf_y_sigmatau
        
        # Return the derivatives of log_f_y with respect to all arguments (y is None since it is not our interest to differentiate in y)
        return None, jac_musigma, jac_sigma2, jac_sigmanu, jac_sigmatau
    
    return first_derivative_sigma, custom_second_derivative

@tf.custom_gradient
def log_pdf_dnu(y, mu, sigma, nu, tau):
    z = boxcox_z(y, mu, sigma, nu)
    w = (tau+1.0)/(tau+z**2.0)
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)

    arg_t = 1.0 / (sigma * tf.math.abs(nu))
    h_sigma_nu_tau = dist.prob( arg_t ) / dist.cdf( arg_t )

    first_derivative_nu = w*z**2.0/nu - (tf.math.log(y)-tf.math.log(mu))*(w*z**2.0 + w*z/(sigma*nu) - 1.0) + tf.math.sign(nu) * h_sigma_nu_tau / (sigma*nu**2.0)

    def custom_second_derivative(upstream_grad):       
        d2log_pdf_y_nu2 = -sigma**2.0*tau*(7.0*tau-9.0) / (4.0*(tau-2.0)*(tau+3.0))
        d2log_pdf_y_munu = (tau-3.0) / (2.0*mu*(tau+3.0))
        d2log_pdf_y_sigmanu = -sigma*nu*tau / (tau+3.0)
        d2log_pdf_y_nutau = sigma**2.0*nu*(2.0*tau+1.0) / ((tau+1.0)*(tau+3.0)*(tau-2.0))

        jac_nu2 = upstream_grad * d2log_pdf_y_nu2
        jac_munu = upstream_grad * d2log_pdf_y_munu
        jac_sigmanu = upstream_grad * d2log_pdf_y_sigmanu
        jac_nutau = upstream_grad * d2log_pdf_y_nutau
        
        # Return the derivatives of log_f_y with respect to all arguments (y is None since it is not our interest to differentiate in y)
        return None, jac_munu, jac_sigmanu, jac_nu2, jac_nutau
    
    return first_derivative_nu, custom_second_derivative

@tf.custom_gradient
def log_pdf_dtau(y, mu, sigma, nu, tau):
    z = boxcox_z(y, mu, sigma, nu)
    w = (tau+1)/(tau+z**2)
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)

    arg_t = 1.0 / (sigma * tf.math.abs(nu))
    h_sigma_nu_tau = dist.prob( arg_t ) / dist.cdf( arg_t )

    # --- j function term estimation from Rigby & Stasinopoulos (2006) - Satistical Modelling using the finite differences method ---
    # Cast variables to float64 to perform numerical differentiation for function j
    tau_64 = tf.cast(tau, tf.float64)
    arg_t_64 = tf.cast(arg_t, tf.float64)
    h = tf.constant(1e-5, dtype = tf.float64)
    
    dist_plus = tfp.distributions.StudentT(df = tau_64 + h, loc = 0.0, scale = 1.0)
    dist_minus = tfp.distributions.StudentT(df = tau_64 - h, loc = 0.0, scale = 1.0)
    
    log_cdf_plus = dist_plus.log_cdf( arg_t_64 )
    log_cdf_minus = dist_minus.log_cdf( arg_t_64 )
    
    j_sigma_nu_tau_64 = (log_cdf_plus - log_cdf_minus) / (2.0 * h)
    
    # Cast function result back to float32
    j_sigma_nu_tau = tf.cast(j_sigma_nu_tau_64, tf.float32)
    # -------------------------------------------------------------------------------------------------------------------------------
    
    first_derivative_tau = -1.0/2.0*tf.math.log(1 + z**2/tau) + w*z**2/(2*tau) + 1.0/2.0 * tf.math.digamma( (tau+1)/2 ) - 1.0/2.0 * tf.math.digamma( tau/2 ) - 1.0/(2.0*tau) - j_sigma_nu_tau

    def custom_second_derivative(upstream_grad):       
        d2log_pdf_y_tau2 = 1.0/4.0*tf.math.polygamma(1.0, (tau+1.0)/2.0) - 1.0/4.0*tf.math.polygamma(1.0, tau/2.0) + (tau+5)/(2*tau*(tau+1)*(tau+3))

        d2log_pdf_y_mutau = 2.0*nu / (mu*(tau+1.0)*(tau+3.0))
        d2log_pdf_y_sigmatau = 2.0 / (sigma*(tau+1.0)*(tau+3.0))
        d2log_pdf_y_nutau = sigma**2.0*nu*(2.0*tau+1.0) / ((tau+1.0)*(tau+3.0)*(tau-2.0))

        jac_tau2 = upstream_grad * d2log_pdf_y_tau2
        jac_mutau = upstream_grad * d2log_pdf_y_mutau
        jac_sigmatau = upstream_grad * d2log_pdf_y_sigmatau
        jac_nutau = upstream_grad * d2log_pdf_y_nutau
        
        # Return the derivatives of log_f_y with respect to all arguments (y is None since it is not our interest to differentiate in y)
        return None, jac_mutau, jac_sigmatau, jac_nutau, jac_tau2
    
    return first_derivative_tau, custom_second_derivative

@tf.custom_gradient
def log_pdf(y, mu, sigma, nu, tau):
    # Enforce float32
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    nu = tf.cast(nu, tf.float32)
    tau = tf.cast(tau, tf.float32)

    epsilon = 1.0e-4
    # Transform y into z using the Box-Cox transformation
    z = boxcox_z(y, mu, sigma, nu, epsilon = epsilon)
    
    # Ensures nu is never properly zero
    safe_nu = tf.where(tf.equal(nu, 0.0), tf.constant(epsilon, dtype = tf.float32), nu)
    
    # Create T distributions with tau degrees of freedom
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)
    
    # We use log_prob directly! Much more stable.
    log_f_z = dist.log_prob(z)

    arg_t = 1.0 / (sigma * tf.math.abs(safe_nu))
    log_cdf_arg_t = dist.log_cdf( arg_t )
    
    # Log-Likelihood value
    log_pdf_y = (safe_nu - 1.0) * tf.math.log(y) - safe_nu*tf.math.log(mu) - tf.math.log(sigma) + log_f_z - log_cdf_arg_t

    def custom_derivative(upstream_grad):
        # We explicitly tells tensorflow the derivative of f(x) = x^3 is a constant function equal to 5        
        dlog_pdf_y_mu = log_pdf_dmu(y, mu, sigma, safe_nu, tau)
        dlog_pdf_y_sigma = log_pdf_dsigma(y, mu, sigma, safe_nu, tau)
        dlog_pdf_y_nu = log_pdf_dnu(y, mu, sigma, safe_nu, tau)
        dlog_pdf_y_tau = log_pdf_dtau(y, mu, sigma, safe_nu, tau)
        
        grad_mu = upstream_grad * dlog_pdf_y_mu
        grad_sigma = upstream_grad * dlog_pdf_y_sigma
        grad_nu = upstream_grad * dlog_pdf_y_nu
        grad_tau = upstream_grad * dlog_pdf_y_tau
        
        # Return the derivatives of log_f_y with respect to all arguments (y is None since it is not our interest to differentiate in y)
        return None, grad_mu, grad_sigma, grad_nu, grad_tau
              
    return log_pdf_y, custom_derivative

@tf.function(reduce_retracing=True)
def pdf(y, mu, sigma, nu, tau):
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    nu = tf.cast(nu, tf.float32)
    tau = tf.cast(tau, tf.float32)

    epsilon = 1.0e-4
    z = boxcox_z(y, mu, sigma, nu, epsilon = epsilon)

    # Ensures nu is never properly zero
    safe_nu = tf.where(tf.equal(nu, 0.0), tf.constant(epsilon, dtype = tf.float32), nu)
    
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)
    arg_t = 1.0 / (sigma * tf.math.abs(safe_nu))
    pdf_z = dist.prob( z ) / dist.cdf( arg_t )
    
    return y**(nu-1)/(mu**nu*sigma) * pdf_z

@tf.function(reduce_retracing=True)
def cdf(y, mu, sigma, nu, tau):
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    nu = tf.cast(nu, tf.float32)
    tau = tf.cast(tau, tf.float32)

    epsilon = 1.0e-4
    z = boxcox_z(y, mu, sigma, nu, epsilon = epsilon)

    # Ensures nu is never properly zero
    safe_nu = tf.where(tf.equal(nu, 0.0), tf.constant(epsilon, dtype = tf.float32), nu)
    
    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)
    arg_t = 1.0 / (sigma * tf.math.abs(safe_nu))

    lower_limit = tf.where(
        safe_nu > 0,
        dist.cdf(-arg_t),
        tf.zeros_like(y)
    )
    
    return (dist.cdf(z) - lower_limit) / dist.cdf(arg_t)

@tf.function(reduce_retracing=True)
def ppf(q, mu, sigma, nu, tau):
    q = tf.cast(q, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    nu = tf.cast(nu, tf.float32)
    tau = tf.cast(tau, tf.float32)
    
    epsilon = 1.0e-4    
    # Tolerance for considering nu to be actually equal to zero
    is_nu_zero = (tf.math.abs(nu) < epsilon)
    
    # If nu is zero, replace it by 1.0e-7 to prevent division by zero
    safe_nu = tf.where(tf.equal(nu, 0.0), tf.constant(epsilon, dtype = tf.float32), nu)

    dist = tfp.distributions.StudentT(df = tau, loc = 0.0, scale = 1.0)
    arg_t = 1.0 / (sigma * tf.math.abs(safe_nu))
    
    z_q = tf.where(
        safe_nu <= 0,
        dist.quantile( q*dist.cdf( arg_t ) ),
        dist.quantile( 1 - (1-q)*dist.cdf( arg_t ) )
    )

    y_q = tf.where(
        is_nu_zero,
        mu * tf.math.exp(sigma * z_q),
        mu * (1 + sigma*safe_nu*z_q)**(1/safe_nu)
    )
    
    return y_q

    
