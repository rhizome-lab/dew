//! Binary and unary operations for complex numbers.

use crate::{ComplexValue, Error, Type};
use num_traits::Float;
use rhizome_dew_core::{BinOp, UnaryOp};

/// Apply a binary operation to two values.
pub fn apply_binop<T, V>(op: BinOp, left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match op {
        BinOp::Add => apply_add(left, right),
        BinOp::Sub => apply_sub(left, right),
        BinOp::Mul => apply_mul(left, right),
        BinOp::Div => apply_div(left, right),
        BinOp::Pow => apply_pow(left, right),
        // Bitwise/modulo operations not supported for complex numbers
        BinOp::Rem | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
            Err(Error::BinaryTypeMismatch {
                op,
                left: left.typ(),
                right: right.typ(),
            })
        }
    }
}

/// Apply a unary operation to a value.
pub fn apply_unaryop<T, V>(op: UnaryOp, val: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match op {
        UnaryOp::Neg => apply_neg(val),
        UnaryOp::Not => apply_not(val),
        // Bitwise NOT not supported for complex numbers
        UnaryOp::BitNot => Err(Error::UnaryTypeMismatch {
            op,
            operand: val.typ(),
        }),
    }
}

// ============================================================================
// Addition
// ============================================================================

fn apply_add<T, V>(left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match (left.typ(), right.typ()) {
        // Scalar + Scalar
        (Type::Scalar, Type::Scalar) => {
            let a = left.as_scalar().unwrap();
            let b = right.as_scalar().unwrap();
            Ok(V::from_scalar(a + b))
        }

        // Complex + Complex
        (Type::Complex, Type::Complex) => {
            let a = left.as_complex().unwrap();
            let b = right.as_complex().unwrap();
            Ok(V::from_complex([a[0] + b[0], a[1] + b[1]]))
        }

        // Scalar + Complex (promote scalar to complex)
        (Type::Scalar, Type::Complex) => {
            let s = left.as_scalar().unwrap();
            let c = right.as_complex().unwrap();
            Ok(V::from_complex([s + c[0], c[1]]))
        }

        // Complex + Scalar
        (Type::Complex, Type::Scalar) => {
            let c = left.as_complex().unwrap();
            let s = right.as_scalar().unwrap();
            Ok(V::from_complex([c[0] + s, c[1]]))
        }
    }
}

// ============================================================================
// Subtraction
// ============================================================================

fn apply_sub<T, V>(left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match (left.typ(), right.typ()) {
        (Type::Scalar, Type::Scalar) => {
            let a = left.as_scalar().unwrap();
            let b = right.as_scalar().unwrap();
            Ok(V::from_scalar(a - b))
        }

        (Type::Complex, Type::Complex) => {
            let a = left.as_complex().unwrap();
            let b = right.as_complex().unwrap();
            Ok(V::from_complex([a[0] - b[0], a[1] - b[1]]))
        }

        (Type::Scalar, Type::Complex) => {
            let s = left.as_scalar().unwrap();
            let c = right.as_complex().unwrap();
            Ok(V::from_complex([s - c[0], -c[1]]))
        }

        (Type::Complex, Type::Scalar) => {
            let c = left.as_complex().unwrap();
            let s = right.as_scalar().unwrap();
            Ok(V::from_complex([c[0] - s, c[1]]))
        }
    }
}

// ============================================================================
// Multiplication
// ============================================================================

fn apply_mul<T, V>(left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match (left.typ(), right.typ()) {
        // Scalar * Scalar
        (Type::Scalar, Type::Scalar) => {
            let a = left.as_scalar().unwrap();
            let b = right.as_scalar().unwrap();
            Ok(V::from_scalar(a * b))
        }

        // Complex * Complex: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        (Type::Complex, Type::Complex) => {
            let a = left.as_complex().unwrap();
            let b = right.as_complex().unwrap();
            let re = a[0] * b[0] - a[1] * b[1];
            let im = a[0] * b[1] + a[1] * b[0];
            Ok(V::from_complex([re, im]))
        }

        // Scalar * Complex
        (Type::Scalar, Type::Complex) => {
            let s = left.as_scalar().unwrap();
            let c = right.as_complex().unwrap();
            Ok(V::from_complex([s * c[0], s * c[1]]))
        }

        // Complex * Scalar
        (Type::Complex, Type::Scalar) => {
            let c = left.as_complex().unwrap();
            let s = right.as_scalar().unwrap();
            Ok(V::from_complex([c[0] * s, c[1] * s]))
        }
    }
}

// ============================================================================
// Division
// ============================================================================

fn apply_div<T, V>(left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match (left.typ(), right.typ()) {
        // Scalar / Scalar
        (Type::Scalar, Type::Scalar) => {
            let a = left.as_scalar().unwrap();
            let b = right.as_scalar().unwrap();
            Ok(V::from_scalar(a / b))
        }

        // Complex / Complex: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
        (Type::Complex, Type::Complex) => {
            let a = left.as_complex().unwrap();
            let b = right.as_complex().unwrap();
            let denom = b[0] * b[0] + b[1] * b[1];
            let re = (a[0] * b[0] + a[1] * b[1]) / denom;
            let im = (a[1] * b[0] - a[0] * b[1]) / denom;
            Ok(V::from_complex([re, im]))
        }

        // Scalar / Complex
        (Type::Scalar, Type::Complex) => {
            // s / (a + bi) = s(a - bi) / (a² + b²)
            let s = left.as_scalar().unwrap();
            let c = right.as_complex().unwrap();
            let denom = c[0] * c[0] + c[1] * c[1];
            let re = s * c[0] / denom;
            let im = -s * c[1] / denom;
            Ok(V::from_complex([re, im]))
        }

        // Complex / Scalar
        (Type::Complex, Type::Scalar) => {
            let c = left.as_complex().unwrap();
            let s = right.as_scalar().unwrap();
            Ok(V::from_complex([c[0] / s, c[1] / s]))
        }
    }
}

// ============================================================================
// Power
// ============================================================================

fn apply_pow<T, V>(left: V, right: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match (left.typ(), right.typ()) {
        // Scalar ^ Scalar
        (Type::Scalar, Type::Scalar) => {
            let a = left.as_scalar().unwrap();
            let b = right.as_scalar().unwrap();
            Ok(V::from_scalar(a.powf(b)))
        }

        // Complex ^ Scalar (integer-ish powers common)
        // z^n = r^n * e^(i*n*θ) where z = r*e^(iθ)
        (Type::Complex, Type::Scalar) => {
            let c = left.as_complex().unwrap();
            let n = right.as_scalar().unwrap();
            let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
            let theta = c[1].atan2(c[0]);
            let r_n = r.powf(n);
            let theta_n = theta * n;
            Ok(V::from_complex([r_n * theta_n.cos(), r_n * theta_n.sin()]))
        }

        // Other cases not supported for now
        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Pow,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Negation
// ============================================================================

fn apply_neg<T, V>(val: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match val.typ() {
        Type::Scalar => {
            let v = val.as_scalar().unwrap();
            Ok(V::from_scalar(-v))
        }
        Type::Complex => {
            let c = val.as_complex().unwrap();
            Ok(V::from_complex([-c[0], -c[1]]))
        }
    }
}

// ============================================================================
// Logical Not
// ============================================================================

fn apply_not<T, V>(val: V) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match val.typ() {
        Type::Scalar => {
            let v = val.as_scalar().unwrap();
            Ok(V::from_scalar(if v == T::zero() {
                T::one()
            } else {
                T::zero()
            }))
        }
        // not doesn't make sense for complex
        Type::Complex => Err(crate::Error::TypeMismatch {
            expected: crate::Type::Scalar,
            got: crate::Type::Complex,
        }),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    #[test]
    fn test_complex_mul() {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let a = Value::Complex([1.0_f32, 2.0]);
        let b = Value::Complex([3.0, 4.0]);
        let result = apply_binop(BinOp::Mul, a, b).unwrap();
        assert_eq!(result, Value::Complex([-5.0, 10.0]));
    }

    #[test]
    fn test_complex_div() {
        // (1+2i) / (1+2i) = 1
        let a = Value::Complex([1.0_f32, 2.0]);
        let b = Value::Complex([1.0, 2.0]);
        let result = apply_binop(BinOp::Div, a, b).unwrap();
        if let Value::Complex(c) = result {
            assert!((c[0] - 1.0).abs() < 0.0001);
            assert!(c[1].abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }

    #[test]
    fn test_complex_pow() {
        // i^2 = -1
        let i = Value::Complex([0.0_f32, 1.0]);
        let two = Value::Scalar(2.0);
        let result = apply_binop(BinOp::Pow, i, two).unwrap();
        if let Value::Complex(c) = result {
            assert!((c[0] - (-1.0)).abs() < 0.0001);
            assert!(c[1].abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }
}
