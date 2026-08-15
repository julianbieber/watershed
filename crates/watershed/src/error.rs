// TODO(jb-doc): why the crate offers one error beside the typed ones rather than instead
// of them, and which callers each spelling is for.

use thiserror::Error;

use crate::bake::{BakeError, PlanError};
use crate::io::IoError;
use crate::water::WaterError;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Bake(#[from] BakeError),
    #[error(transparent)]
    Water(#[from] WaterError),
}
