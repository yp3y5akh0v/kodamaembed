use super::{Activation, Error, Weight, check_dot, is_supported, validate_depth};

#[test]
fn invalid_shapes_ranges_and_mismatched_depth_are_rejected() {
    if !is_supported() {
        return;
    }
    assert_eq!(validate_depth(0), Err(Error::Dimensions));
    assert!(matches!(Weight::new(&[], 0, 1), Err(Error::Dimensions)));
    assert!(matches!(Weight::new(&[0], 1, 0), Err(Error::Dimensions)));
    assert!(matches!(
        Weight::new(&[], usize::MAX, 1),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        Weight::new(&[i8::MIN], 1, 1),
        Err(Error::IntegerRange)
    ));
    assert!(matches!(
        Activation::new(&[i8::MIN], 1, 1),
        Err(Error::IntegerRange)
    ));
    assert!(matches!(
        Activation::new(&[], usize::MAX, 3),
        Err(Error::Dimensions)
    ));
    assert!(matches!(
        Activation::new(&[0], 2, 1),
        Err(Error::Dimensions)
    ));
    // Neither a depth the weight's does not divide nor digits that split a group of four.
    let a = Activation::new(&[1, 2, 3], 1, 3).unwrap();
    let w = Weight::new(&[1, 2], 1, 2).unwrap();
    assert_eq!(check_dot(&a, &w), Err(Error::Dimensions));
    let a = Activation::new(&[1, 2], 1, 2).unwrap();
    let w = Weight::new(&[1], 1, 1).unwrap();
    assert_eq!(check_dot(&a, &w), Err(Error::Dimensions));
}
