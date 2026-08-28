//! The one error type a caller mixing several of this crate's operations can hold.

use thiserror::Error;

use crate::bake::{BakeError, PlanError};
use crate::io::IoError;
use crate::water::WaterError;

/// Any failure this crate can produce, as a single type.
///
/// Nothing here returns it — every operation returns its own narrower error, so a
/// caller handling one kind of failure never has to match on variants it cannot
/// reach. This exists for the caller that does several of them in one function and
/// wants `?` to work across all of them; the `From` impls make that conversion.
///
/// Displays as the underlying error verbatim, so wrapping adds no prefix.
#[derive(Debug, Error)]
pub enum Error {
    /// Reading or writing a document. See [`IoError`].
    #[error(transparent)]
    Io(#[from] IoError),
    /// A document that cannot be turned into a bake plan. See [`PlanError`].
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// A bake driven out of order. See [`BakeError`].
    #[error(transparent)]
    Bake(#[from] BakeError),
    /// The water solve. See [`WaterError`].
    #[error(transparent)]
    Water(#[from] WaterError),
}
