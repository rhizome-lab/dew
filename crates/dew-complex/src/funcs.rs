//! Complex number functions: conj, abs, arg, exp, log, etc.

use crate::{ComplexFn, ComplexValue, FunctionRegistry, Signature, Type, Value};
use num_traits::Float;

// ============================================================================
// Real part
// ============================================================================

/// Extract real part: re(z) -> scalar
pub struct Re;

impl<T, V> ComplexFn<T, V> for Re
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "re"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Complex],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let c = args[0].as_complex().unwrap();
        V::from_scalar(c[0])
    }
}

// ============================================================================
// Imaginary part
// ============================================================================

/// Extract imaginary part: im(z) -> scalar
pub struct Im;

impl<T, V> ComplexFn<T, V> for Im
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "im"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Complex],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let c = args[0].as_complex().unwrap();
        V::from_scalar(c[1])
    }
}

// ============================================================================
// Conjugate
// ============================================================================

/// Complex conjugate: conj(a + bi) = a - bi
pub struct Conj;

impl<T, V> ComplexFn<T, V> for Conj
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "conj"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Complex],
            ret: Type::Complex,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let c = args[0].as_complex().unwrap();
        V::from_complex([c[0], -c[1]])
    }
}

// ============================================================================
// Absolute value (magnitude)
// ============================================================================

/// Magnitude: abs(z) = sqrt(re² + im²)
pub struct Abs;

impl<T, V> ComplexFn<T, V> for Abs
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "abs"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Scalar],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Complex],
                ret: Type::Scalar,
            },
        ]
    }

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Scalar => {
                let s = args[0].as_scalar().unwrap();
                V::from_scalar(s.abs())
            }
            Type::Complex => {
                let c = args[0].as_complex().unwrap();
                V::from_scalar((c[0] * c[0] + c[1] * c[1]).sqrt())
            }
        }
    }
}

// ============================================================================
// Argument (phase angle)
// ============================================================================

/// Phase angle: arg(z) = atan2(im, re)
pub struct Arg;

impl<T, V> ComplexFn<T, V> for Arg
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "arg"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Complex],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let c = args[0].as_complex().unwrap();
        V::from_scalar(c[1].atan2(c[0]))
    }
}

// ============================================================================
// Norm (squared magnitude)
// ============================================================================

/// Squared magnitude: norm(z) = re² + im²
pub struct Norm;

impl<T, V> ComplexFn<T, V> for Norm
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "norm"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Complex],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let c = args[0].as_complex().unwrap();
        V::from_scalar(c[0] * c[0] + c[1] * c[1])
    }
}

// ============================================================================
// Complex exponential
// ============================================================================

/// Complex exponential: exp(a + bi) = e^a * (cos(b) + i*sin(b))
pub struct Exp;

impl<T, V> ComplexFn<T, V> for Exp
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "exp"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Scalar],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Complex],
                ret: Type::Complex,
            },
        ]
    }

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Scalar => {
                let s = args[0].as_scalar().unwrap();
                V::from_scalar(s.exp())
            }
            Type::Complex => {
                let c = args[0].as_complex().unwrap();
                let e_a = c[0].exp();
                V::from_complex([e_a * c[1].cos(), e_a * c[1].sin()])
            }
        }
    }
}

// ============================================================================
// Complex logarithm
// ============================================================================

/// Complex logarithm: log(z) = ln|z| + i*arg(z)
pub struct Log;

impl<T, V> ComplexFn<T, V> for Log
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "log"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Scalar],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Complex],
                ret: Type::Complex,
            },
        ]
    }

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Scalar => {
                let s = args[0].as_scalar().unwrap();
                V::from_scalar(s.ln())
            }
            Type::Complex => {
                let c = args[0].as_complex().unwrap();
                let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
                let theta = c[1].atan2(c[0]);
                V::from_complex([r.ln(), theta])
            }
        }
    }
}

// ============================================================================
// Complex square root
// ============================================================================

/// Complex square root
pub struct Sqrt;

impl<T, V> ComplexFn<T, V> for Sqrt
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "sqrt"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Scalar],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Complex],
                ret: Type::Complex,
            },
        ]
    }

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Scalar => {
                let s = args[0].as_scalar().unwrap();
                V::from_scalar(s.sqrt())
            }
            Type::Complex => {
                // sqrt(z) = sqrt(r) * e^(i*θ/2)
                let c = args[0].as_complex().unwrap();
                let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
                let theta = c[1].atan2(c[0]);
                let sqrt_r = r.sqrt();
                let half_theta = theta / T::from(2.0).unwrap();
                V::from_complex([sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin()])
            }
        }
    }
}

// ============================================================================
// Complex power
// ============================================================================

/// Complex power: pow(z, w) = exp(w * log(z))
pub struct Pow;

impl<T, V> ComplexFn<T, V> for Pow
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "pow"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Scalar, Type::Scalar],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Complex, Type::Scalar],
                ret: Type::Complex,
            },
            Signature {
                args: vec![Type::Complex, Type::Complex],
                ret: Type::Complex,
            },
        ]
    }

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Scalar, Type::Scalar) => {
                let a = args[0].as_scalar().unwrap();
                let b = args[1].as_scalar().unwrap();
                V::from_scalar(a.powf(b))
            }
            (Type::Complex, Type::Scalar) => {
                // z^n = r^n * e^(i*n*θ)
                let z = args[0].as_complex().unwrap();
                let n = args[1].as_scalar().unwrap();
                let r = (z[0] * z[0] + z[1] * z[1]).sqrt();
                let theta = z[1].atan2(z[0]);
                let r_n = r.powf(n);
                let theta_n = theta * n;
                V::from_complex([r_n * theta_n.cos(), r_n * theta_n.sin()])
            }
            (Type::Complex, Type::Complex) => {
                // z^w = exp(w * log(z))
                let z = args[0].as_complex().unwrap();
                let w = args[1].as_complex().unwrap();
                let r = (z[0] * z[0] + z[1] * z[1]).sqrt();
                let theta = z[1].atan2(z[0]);
                let ln_r = r.ln();
                // w * log(z) = (w_re + w_im*i) * (ln_r + theta*i)
                // = w_re*ln_r - w_im*theta + i*(w_re*theta + w_im*ln_r)
                let re = w[0] * ln_r - w[1] * theta;
                let im = w[0] * theta + w[1] * ln_r;
                // exp(re + im*i)
                let e_re = re.exp();
                V::from_complex([e_re * im.cos(), e_re * im.sin()])
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Polar construction
// ============================================================================

/// Construct from polar: polar(r, θ) = r * e^(iθ) = r*cos(θ) + i*r*sin(θ)
pub struct Polar;

impl<T, V> ComplexFn<T, V> for Polar
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "polar"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar],
            ret: Type::Complex,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let r = args[0].as_scalar().unwrap();
        let theta = args[1].as_scalar().unwrap();
        V::from_complex([r * theta.cos(), r * theta.sin()])
    }
}

// ============================================================================
// Cartesian construction
// ============================================================================

/// Construct from cartesian: complex(re, im) = re + im*i
pub struct Complex;

impl<T, V> ComplexFn<T, V> for Complex
where
    T: Float,
    V: ComplexValue<T>,
{
    fn name(&self) -> &str {
        "complex"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar],
            ret: Type::Complex,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let re = args[0].as_scalar().unwrap();
        let im = args[1].as_scalar().unwrap();
        V::from_complex([re, im])
    }
}

// ============================================================================
// Registry helper
// ============================================================================

/// Register all standard complex functions.
pub fn register_complex<T, V>(registry: &mut FunctionRegistry<T, V>)
where
    T: Float + 'static,
    V: ComplexValue<T> + 'static,
{
    registry.register(Re);
    registry.register(Im);
    registry.register(Conj);
    registry.register(Abs);
    registry.register(Arg);
    registry.register(Norm);
    registry.register(Exp);
    registry.register(Log);
    registry.register(Sqrt);
    registry.register(Pow);
    registry.register(Polar);
    registry.register(Complex);
}

/// Create a new registry with all standard complex functions.
pub fn complex_registry<T: Float + 'static>() -> FunctionRegistry<T, Value<T>>
where
    T: std::fmt::Debug,
{
    let mut registry = FunctionRegistry::new();
    register_complex(&mut registry);
    registry
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;
    use std::collections::HashMap;

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Value<f32> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = complex_registry();
        crate::eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_re_im() {
        let result = eval_expr("re(z)", &[("z", Value::Complex([3.0, 4.0]))]);
        assert_eq!(result, Value::Scalar(3.0));

        let result = eval_expr("im(z)", &[("z", Value::Complex([3.0, 4.0]))]);
        assert_eq!(result, Value::Scalar(4.0));
    }

    #[test]
    fn test_conj() {
        let result = eval_expr("conj(z)", &[("z", Value::Complex([3.0, 4.0]))]);
        assert_eq!(result, Value::Complex([3.0, -4.0]));
    }

    #[test]
    fn test_abs() {
        // |3 + 4i| = 5
        let result = eval_expr("abs(z)", &[("z", Value::Complex([3.0, 4.0]))]);
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_arg() {
        // arg(1 + i) = π/4
        let result = eval_expr("arg(z)", &[("z", Value::Complex([1.0, 1.0]))]);
        if let Value::Scalar(v) = result {
            assert!((v - std::f32::consts::FRAC_PI_4).abs() < 0.0001);
        } else {
            panic!("expected scalar");
        }
    }

    #[test]
    fn test_exp() {
        // exp(iπ) = -1
        let result = eval_expr(
            "exp(z)",
            &[("z", Value::Complex([0.0, std::f32::consts::PI]))],
        );
        if let Value::Complex(c) = result {
            assert!((c[0] - (-1.0)).abs() < 0.0001);
            assert!(c[1].abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }

    #[test]
    fn test_polar() {
        // polar(1, π/2) = i
        let result = eval_expr(
            "polar(r, theta)",
            &[
                ("r", Value::Scalar(1.0)),
                ("theta", Value::Scalar(std::f32::consts::FRAC_PI_2)),
            ],
        );
        if let Value::Complex(c) = result {
            assert!(c[0].abs() < 0.0001);
            assert!((c[1] - 1.0).abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }

    #[test]
    fn test_sqrt() {
        // sqrt(-1) = i
        let result = eval_expr("sqrt(z)", &[("z", Value::Complex([-1.0, 0.0]))]);
        if let Value::Complex(c) = result {
            assert!(c[0].abs() < 0.0001);
            assert!((c[1] - 1.0).abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }

    #[test]
    fn test_complex() {
        // complex(3, 4) = 3 + 4i
        let result = eval_expr(
            "complex(re, im)",
            &[("re", Value::Scalar(3.0)), ("im", Value::Scalar(4.0))],
        );
        assert_eq!(result, Value::Complex([3.0, 4.0]));
    }
}
