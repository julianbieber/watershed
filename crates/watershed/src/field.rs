//! How a field is named and what it is for, independently of anything that
//! produces its values.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The name a field is addressed by, everywhere: in a document, in a layer that
/// references another field, and in [`Terrain`](crate::terrain::Terrain) lookups.
///
/// Serialized as a plain string, so it is what a person editing a document by hand
/// sees and types. Any string is accepted here; uniqueness within a document is a
/// check made where a document is planned, and a reference to a name no field
/// carries fails there too rather than at lookup.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldId(String);

impl FieldId {
    /// Takes the name verbatim — no trimming, casing or validation. Two ids are
    /// equal exactly when their strings are.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name, as it appears in a serialized document.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for FieldId {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl From<String> for FieldId {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl AsRef<str> for FieldId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What a field *is*, independently of what it is called, so that a consumer can
/// find the height of a document it did not author.
///
/// A document may hold at most one `Height` and at most one `Moisture` field, and
/// the `Height` field must be at shift 0. Neither is enforced here — both are
/// checked where a document is planned. `Custom` carries no such constraint and
/// any number of fields may hold it, which is why looking a `Custom` field up by
/// role never resolves.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum FieldRole {
    /// The elevation of the document. A water solve cannot be planned without one,
    /// and it must be at shift 0.
    Height,
    /// Wetness of the document. Carries no constraint beyond uniqueness — nothing
    /// in this crate reads a field *because* it holds this role.
    Moisture,
    /// No bake meaning at all. The default, so a field acquires a role only by
    /// being given one.
    #[default]
    Custom,
}

impl FieldRole {
    /// Every role, in the order the editor offers them.
    pub const ALL: [Self; 3] = [Self::Height, Self::Moisture, Self::Custom];

    /// The lowercase word this role serializes and displays as, and the only
    /// spelling [`FieldRole::parse`] accepts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Height => "height",
            Self::Moisture => "moisture",
            Self::Custom => "custom",
        }
    }

    /// The role spelled exactly as [`FieldRole::as_str`] writes it, or `None`.
    /// Case-sensitive, and does not trim.
    pub fn parse(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|role| role.as_str() == word)
    }
}

impl fmt::Display for FieldRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // Four constructors reach the same string and equality is by string; a
    // conversion that trimmed or cased would make two spellings of a name address
    // different fields.
    #[test]
    fn a_field_id_round_trips_through_every_way_of_making_one() {
        assert_eq!(FieldId::from("height").as_str(), "height");
        assert_eq!(FieldId::new("height"), FieldId::from("height".to_owned()));
        assert_eq!(FieldId::from("height").to_string(), "height");
    }
}
