#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

//! Session lifecycle model with deterministic expiration helpers.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

/// Errors that may arise when constructing or updating a [`Session`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionValidationError {
    /// The provided time-to-live would create an immediately expired session.
    NonPositiveTtl,

    /// The supplied expiration precedes the creation timestamp.
    ExpirationBeforeCreation,
}

impl fmt::Display for SessionValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveTtl => write!(f, "session TTL must be positive"),
            Self::ExpirationBeforeCreation => {
                write!(f, "expiration timestamp must be after creation time")
            }
        }
    }
}

impl Error for SessionValidationError {}

/// A session record persisted in storage with expiration semantics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Session {
    /// Unique identifier for this session instance.
    pub session_id: Uuid,
    /// Identifier of the authenticated principal owning the session.
    pub user_id: Uuid,
    /// UTC timestamp representing when the session was created.
    pub created_at: DateTime<Utc>,
    /// UTC timestamp after which the session is no longer considered valid.
    pub expires_at: DateTime<Utc>,
}

impl Session {
    /// Create a new session with a freshly generated identifier and time-to-live.
    ///
    /// # Errors
    ///
    /// Returns [`SessionValidationError::NonPositiveTtl`] if `ttl` is zero or negative.
    #[must_use]
    pub fn create(user_id: Uuid, ttl: Duration) -> Result<Self, SessionValidationError> {
        let created_at = Utc::now();
        Self::from_parts(Uuid::new_v4(), user_id, created_at, created_at + ttl)
    }

    /// Construct a session from explicit parts, validating its invariants.
    ///
    /// # Errors
    ///
    /// Returns [`SessionValidationError::ExpirationBeforeCreation`] if `expires_at`
    /// is not strictly greater than `created_at`.
    pub fn from_parts(
        session_id: Uuid,
        user_id: Uuid,
        created_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
    ) -> Result<Self, SessionValidationError> {
        if expires_at <= created_at {
            return Err(SessionValidationError::ExpirationBeforeCreation);
        }

        Ok(Self {
            session_id,
            user_id,
            created_at,
            expires_at,
        })
    }

    /// Returns `true` if the session has expired relative to the current wall clock.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(Utc::now())
    }

    /// Returns `true` if the session has expired relative to a provided timestamp.
    #[must_use]
    pub fn is_expired_at(&self, now: DateTime<Utc>) -> bool {
        now >= self.expires_at
    }

    /// Remaining time before the session expires, saturating at zero.
    #[must_use]
    pub fn remaining_time(&self, now: DateTime<Utc>) -> Duration {
        if now >= self.expires_at {
            Duration::zero()
        } else {
            self.expires_at - now
        }
    }

    /// Extend the session lifetime by the provided positive duration.
    ///
    /// # Errors
    ///
    /// Returns [`SessionValidationError::NonPositiveTtl`] when `ttl` is zero or negative.
    pub fn renew(&mut self, ttl: Duration) -> Result<(), SessionValidationError> {
        if ttl <= Duration::zero() {
            return Err(SessionValidationError::NonPositiveTtl);
        }

        let now = Utc::now();
        let base = if now > self.expires_at {
            now
        } else {
            self.expires_at
        };
        self.expires_at = base + ttl;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Session, SessionValidationError};
    use chrono::{Duration, TimeZone, Utc};
    use serde_json::json;
    use std::mem;
    use uuid::Uuid;

    #[test]
    fn detects_expiration_and_activity() {
        let created = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let expires = created + Duration::minutes(5);
        let session = Session::from_parts(Uuid::nil(), Uuid::new_v4(), created, expires)
            .expect("valid session");

        let before_expiry = created + Duration::minutes(3);
        assert!(!session.is_expired_at(before_expiry));
        assert_eq!(session.remaining_time(before_expiry), Duration::minutes(2));

        let after_expiry = expires + Duration::seconds(1);
        assert!(session.is_expired_at(after_expiry));
        assert_eq!(session.remaining_time(after_expiry), Duration::zero());
    }

    #[test]
    fn rejects_invalid_configurations() {
        let created = Utc.timestamp_opt(1_700_000_000, 0).single().unwrap();
        let expires = created;
        let err = Session::from_parts(Uuid::new_v4(), Uuid::new_v4(), created, expires)
            .expect_err("expiration must be after creation");
        assert_eq!(err, SessionValidationError::ExpirationBeforeCreation);

        let mut session = Session::from_parts(
            Uuid::new_v4(),
            Uuid::new_v4(),
            created,
            created + Duration::hours(1),
        )
        .unwrap();

        let renew_err = session.renew(Duration::zero()).expect_err("renewal ttl");
        assert_eq!(renew_err, SessionValidationError::NonPositiveTtl);
    }

    #[test]
    fn serializes_with_uuid_support() {
        let created = Utc
            .timestamp_opt(1_700_000_000, 500_000_000)
            .single()
            .unwrap();
        let expires = created + Duration::hours(2);
        let session = Session::from_parts(
            Uuid::parse_str("01020304-0506-0708-090a-0b0c0d0e0f10").unwrap(),
            Uuid::parse_str("ffffffff-ffff-ffff-ffff-ffffffffffff").unwrap(),
            created,
            expires,
        )
        .unwrap();

        let json_value = serde_json::to_value(&session).expect("serialize session");
        assert_eq!(
            json_value,
            json!({
                "session_id": "01020304-0506-0708-090a-0b0c0d0e0f10",
                "user_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
                "created_at": "2023-11-14T22:13:20.500Z",
                "expires_at": "2023-11-15T00:13:20.500Z"
            })
        );

        let round_trip: Session = serde_json::from_value(json_value).expect("deserialize session");
        assert_eq!(round_trip, session);
    }

    #[test]
    fn memory_layout_matches_component_sum() {
        let expected = mem::size_of::<Uuid>() * 2 + mem::size_of::<chrono::DateTime<Utc>>() * 2;
        assert_eq!(mem::size_of::<Session>(), expected);
    }
}
